"""
T3 (Token-to-Token) TTS model.

Main model that converts text tokens to speech tokens.
"""

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from .config import T3Config
from .backbone import LLaMABackbone
from .conditioning import T3Cond, T3CondEnc
from .sampling import process_logits, sample_token

logger = logging.getLogger(__name__)


class LearnedPositionEmbeddings(nn.Module):
    """
    Learned position embeddings for text and speech tokens.
    """

    def __init__(self, max_seq_len: int, dim: int, init_std: float = 0.02):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(max_seq_len, dim))
        nn.init.normal_(self.weight, mean=0.0, std=init_std)

    def forward(self, x: Tensor) -> Tensor:
        """Get position embeddings for sequence length of x."""
        seq_len = x.shape[1]
        return self.weight[:seq_len]

    def get_fixed_embedding(self, idx: int) -> Tensor:
        """Get embedding at a specific position."""
        return self.weight[idx : idx + 1].unsqueeze(0)


class T3(nn.Module):
    """
    Token-to-Token TTS model.

    Converts text tokens to speech tokens using a LLaMA-style transformer
    with conditioning from speaker embeddings and optional speech prompts.

    Args:
        config: T3 configuration
    """

    def __init__(self, config: Optional[T3Config] = None):
        super().__init__()

        if config is None:
            config = T3Config.english_only()

        self.config = config
        self.hp = config  # Alias for compatibility

        # Embeddings
        self.text_emb = nn.Embedding(config.text_tokens_dict_size, config.hidden_size)
        self.speech_emb = nn.Embedding(config.speech_tokens_dict_size, config.hidden_size)

        # Position embeddings (optional)
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if config.input_pos_emb == "learned":
            max_text_len = config.max_text_tokens + 2
            max_speech_len = config.max_speech_tokens + 4
            self.text_pos_emb = LearnedPositionEmbeddings(max_text_len, config.hidden_size)
            self.speech_pos_emb = LearnedPositionEmbeddings(max_speech_len, config.hidden_size)

        # Conditioning encoder
        self.cond_enc = T3CondEnc(config)

        # Transformer backbone
        self.backbone = LLaMABackbone(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            rope_scaling_factor=config.rope_scaling_factor,
            rope_high_freq_factor=config.rope_high_freq_factor,
            rope_low_freq_factor=config.rope_low_freq_factor,
            rope_original_max_position_embeddings=config.rope_original_max_position_embeddings,
        )

        # Output heads (no bias to match checkpoint)
        self.text_head = nn.Linear(config.hidden_size, config.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(config.hidden_size, config.speech_tokens_dict_size, bias=False)

    @property
    def device(self) -> torch.device:
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond) -> Tensor:
        """
        Prepare conditioning embeddings.

        If speech prompt tokens are provided but not yet embedded,
        embed them using the speech embedding layer.
        """
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            # Embed speech prompt tokens
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)

            # Add position embeddings if using learned positions
            if self.speech_pos_emb is not None:
                t3_cond.cond_prompt_speech_emb = (
                    t3_cond.cond_prompt_speech_emb + self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
                )

        return self.cond_enc(t3_cond)

    def prepare_input_embeds(
        self,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        speech_tokens: Tensor,
        cfg_weight: float = 0.0,
    ) -> Tuple[Tensor, int]:
        """
        Prepare input embeddings for the transformer.

        Concatenates: [conditioning] + [text] + [speech]

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs
            speech_tokens: Speech token IDs
            cfg_weight: CFG weight (if > 0, second batch item is unconditioned)

        Returns:
            Tuple of (embeddings, conditioning_length)
        """
        # Get conditioning embeddings
        cond_emb = self.prepare_conditioning(t3_cond)  # (B, cond_len, dim)

        # Embed text tokens
        text_emb = self.text_emb(text_tokens)  # (B, text_len, dim)

        # Zero out text content for CFG unconditional BEFORE position embeddings
        # This way, unconditioned batch still has position information
        if cfg_weight > 0.0 and text_emb.shape[0] > 1:
            text_emb = text_emb.clone()  # Ensure we can modify in-place
            text_emb[1].zero_()

        # Add position embeddings (after zeroing text content)
        if self.text_pos_emb is not None:
            text_emb = text_emb + self.text_pos_emb(text_tokens)

        # Embed speech tokens
        speech_emb = self.speech_emb(speech_tokens)  # (B, speech_len, dim)

        # Add position embeddings for speech
        if self.speech_pos_emb is not None:
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)

        cond_len = cond_emb.shape[1]

        # Expand tensors if batch sizes don't match (for CFG)
        batch_size = text_emb.shape[0]
        if cond_emb.shape[0] != batch_size:
            cond_emb = cond_emb.expand(batch_size, -1, -1)
        if speech_emb.shape[0] != batch_size:
            speech_emb = speech_emb.expand(batch_size, -1, -1)

        # Concatenate: [cond, text, speech]
        embeds = torch.cat([cond_emb, text_emb, speech_emb], dim=1)

        return embeds, cond_len

    def forward(
        self,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        text_token_lens: Tensor,
        speech_tokens: Tensor,
        speech_token_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass for training.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (B, text_len)
            text_token_lens: Actual lengths of text sequences
            speech_tokens: Speech token IDs (B, speech_len)
            speech_token_lens: Actual lengths of speech sequences

        Returns:
            Tuple of (text_logits, speech_logits)
        """
        # Prepare embeddings
        embeds, cond_len = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=speech_tokens,
        )

        # Run through backbone
        hidden_states, _ = self.backbone(inputs_embeds=embeds)

        # Extract text and speech hidden states
        text_len = text_tokens.shape[1]
        speech_len = speech_tokens.shape[1]
        B, _, dim = hidden_states.shape

        text_start = cond_len
        text_end = cond_len + text_len
        speech_start = text_end
        speech_end = speech_start + speech_len

        text_hidden = hidden_states[:, text_start:text_end, :]
        speech_hidden = hidden_states[:, speech_start:speech_end, :]

        # Project to logits
        text_logits = self.text_head(text_hidden)
        speech_logits = self.speech_head(speech_hidden)

        return text_logits, speech_logits

    @torch.inference_mode()
    def inference(
        self,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
        use_alignment_analyzer: bool = False,
        token_repetition_threshold: int = 3,
        trim_buffer: int = 25,
    ) -> Tensor:
        """
        Generate speech tokens autoregressively.

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs (1, text_len) or (2, text_len) for CFG
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            min_p: Min-p sampling parameter
            repetition_penalty: Repetition penalty factor
            cfg_weight: Classifier-free guidance weight

        Returns:
            Generated speech tokens (1, generated_len)
        """
        text_tokens = torch.atleast_2d(text_tokens).to(self.device)
        batch_size = text_tokens.shape[0]  # Will be 2 if CFG is used (caller duplicates)

        # Initialize with start-of-speech token
        sos_token = torch.tensor(
            [[self.config.start_speech_token]], device=self.device, dtype=torch.long
        )

        # Duplicate sos_token to match text_tokens batch size
        if batch_size > 1:
            sos_token = sos_token.expand(batch_size, -1)

        # Prepare initial embeddings (contains cond + text + initial speech token)
        # Note: caller (tts.py) already duplicates text_tokens for CFG
        # prepare_input_embeds will zero text_emb[1] for unconditioned batch
        embeds, cond_len = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=sos_token,
            cfg_weight=cfg_weight,
        )

        # Create BOS embed to append (matching original behavior)
        # Original concatenates embeds with an additional bos_embed at the end
        bos_token_single = torch.tensor(
            [[self.config.start_speech_token]], device=self.device, dtype=torch.long
        )
        bos_embed = self.speech_emb(bos_token_single)
        if self.speech_pos_emb is not None:
            bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)

        # Duplicate for CFG batch
        if batch_size > 1:
            bos_embed = bos_embed.expand(batch_size, -1, -1)

        # Concatenate: [cond, text, speech_start, bos_start]
        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

        # Track generated tokens (only conditioned batch)
        sos_token_single = torch.tensor(
            [[self.config.start_speech_token]], device=self.device, dtype=torch.long
        )
        generated_tokens = [sos_token_single]

        # Initial forward pass (no cache)
        hidden_states, past_key_values = self.backbone(
            inputs_embeds=inputs_embeds,
            use_cache=True,
        )

        # Create alignment analyzer for hallucination detection
        analyzer = None
        if use_alignment_analyzer:
            from .alignment import AlignmentStreamAnalyzer
            text_len = text_tokens.shape[1]
            text_tokens_slice = (cond_len, cond_len + text_len)
            analyzer = AlignmentStreamAnalyzer(
                self.backbone,
                text_tokens_slice=text_tokens_slice,
                eos_idx=self.config.stop_speech_token,
                token_repetition_threshold=token_repetition_threshold,
                trim_buffer=trim_buffer,
            )

        # Generation loop
        for step in tqdm(range(max_new_tokens), desc="Generating"):
            # Get logits from last position
            logits = self.speech_head(hidden_states[:, -1:, :]).squeeze(1)  # (B, vocab)

            # Apply CFG
            if cfg_weight > 0.0:
                cond_logits = logits[0:1]
                uncond_logits = logits[1:2]
                logits = cond_logits + cfg_weight * (cond_logits - uncond_logits)
            else:
                logits = logits[0:1]

            # Alignment analysis (hallucination detection)
            if analyzer is not None:
                last_token = generated_tokens[-1].item() if generated_tokens else None
                logits = analyzer.step(logits, next_token=last_token)

            # Process logits
            input_ids = torch.cat(generated_tokens, dim=1)
            logits = process_logits(
                logits,
                input_ids=input_ids,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )

            # Sample next token
            next_token = sample_token(logits, do_sample=True)
            generated_tokens.append(next_token)

            # Check for EOS
            if next_token.item() == self.config.stop_speech_token:
                break

            # Prepare next input
            next_emb = self.speech_emb(next_token)
            if self.speech_pos_emb is not None:
                next_emb = next_emb + self.speech_pos_emb.get_fixed_embedding(step + 1)

            # Duplicate for CFG
            if cfg_weight > 0.0:
                next_emb = torch.cat([next_emb, next_emb], dim=0)

            # Forward with cache
            hidden_states, past_key_values = self.backbone(
                inputs_embeds=next_emb,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # Concatenate all generated tokens
        all_tokens = torch.cat(generated_tokens, dim=1)

        # Trim garbage tail if alignment analyzer detected long_tail/alignment_repetition
        # trim_to is in frame positions; +1 accounts for the initial SOS token
        if analyzer is not None and analyzer.trim_to is not None:
            trim_len = analyzer.trim_to + 1
            if trim_len < all_tokens.shape[1]:
                logger.info(f"Trimming speech tokens from {all_tokens.shape[1]} to {trim_len}")
                all_tokens = all_tokens[:, :trim_len]

        # Store alignment analyzer state (for retry logic)
        # forced_eos: analyzer forced early stop
        # text_complete: text was fully spoken before forced EOS
        if analyzer is not None:
            self._last_forced_eos = analyzer.forced_eos
            self._last_text_complete = analyzer.complete
        else:
            self._last_forced_eos = False
            self._last_text_complete = True

        return all_tokens

    @torch.inference_mode()
    def inference_batch(
        self,
        t3_cond: T3Cond,
        text_tokens_list: List[Tensor],
        max_new_tokens: int = 1000,
        temperature: float = 0.8,
        top_p: float = 0.95,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
        use_alignment_analyzer: bool = False,
        token_repetition_threshold: int = 3,
        trim_buffer: int = 25,
    ) -> List[Tensor]:
        """
        Batched speech token generation for multiple text sequences in parallel.

        Each text sequence is processed simultaneously on the GPU.
        Uses left-padding so all sequences align at the right edge.

        Args:
            t3_cond: Conditioning data (shared across all sequences)
            text_tokens_list: List of N text token tensors, each (2, text_len_i) for CFG
            max_new_tokens: Maximum tokens to generate per sequence
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            min_p: Min-p sampling parameter
            repetition_penalty: Repetition penalty factor
            cfg_weight: Classifier-free guidance weight
            use_alignment_analyzer: Enable hallucination detection
            token_repetition_threshold: Consecutive identical tokens to trigger forced EOS
            trim_buffer: Frames to keep after text completion when trimming

        Returns:
            List of N generated speech token tensors
        """
        N = len(text_tokens_list)
        device = self.device

        sos_token = torch.tensor(
            [[self.config.start_speech_token]], device=device, dtype=torch.long
        )

        # 1. Prepare embeddings for each sequence (with CFG doubling)
        all_embeds = []
        all_cond_lens = []
        all_text_lens = []
        for text_tokens in text_tokens_list:
            text_tokens = torch.atleast_2d(text_tokens).to(device)
            batch_size_i = text_tokens.shape[0]  # 2 for CFG
            sos_i = sos_token.expand(batch_size_i, -1)

            embeds, cond_len = self.prepare_input_embeds(
                t3_cond=t3_cond,
                text_tokens=text_tokens,
                speech_tokens=sos_i,
                cfg_weight=cfg_weight,
            )

            # Add BOS embed
            bos_token_single = torch.tensor(
                [[self.config.start_speech_token]], device=device, dtype=torch.long
            )
            bos_embed = self.speech_emb(bos_token_single)
            if self.speech_pos_emb is not None:
                bos_embed = bos_embed + self.speech_pos_emb.get_fixed_embedding(0)
            if batch_size_i > 1:
                bos_embed = bos_embed.expand(batch_size_i, -1, -1)

            embeds = torch.cat([embeds, bos_embed], dim=1)  # (2, real_len, hidden)
            all_embeds.append(embeds)
            all_cond_lens.append(cond_len)
            all_text_lens.append(text_tokens.shape[1])

        # 2. Left-pad and stack into a single batch
        real_lens = [e.shape[1] for e in all_embeds]
        max_len = max(real_lens)
        pad_lens = [max_len - rl for rl in real_lens]
        hidden_dim = all_embeds[0].shape[2]

        padded = []
        for embeds, pad_len in zip(all_embeds, pad_lens):
            if pad_len > 0:
                pad = torch.zeros(embeds.shape[0], pad_len, hidden_dim, device=device)
                embeds = torch.cat([pad, embeds], dim=1)
            padded.append(embeds)

        # (2*N, max_len, hidden) — interleaved [cond_0, uncond_0, cond_1, uncond_1, ...]
        inputs_embeds = torch.cat(padded, dim=0)
        total_batch = inputs_embeds.shape[0]  # 2*N

        # 3. Create attention mask (causal + padding)
        causal_mask = torch.triu(
            torch.full((max_len, max_len), float("-inf"), device=device),
            diagonal=1,
        )
        padding_mask = torch.zeros(total_batch, 1, 1, max_len, device=device)
        for i in range(N):
            pl = pad_lens[i]
            if pl > 0:
                padding_mask[2 * i, :, :, :pl] = float("-inf")
                padding_mask[2 * i + 1, :, :, :pl] = float("-inf")
        attention_mask = causal_mask.unsqueeze(0).unsqueeze(0) + padding_mask

        # Fix: allow padding positions to self-attend to avoid NaN from softmax(all -inf).
        # Without this, padding query positions have ALL -inf attention weights (causal blocks
        # future, padding mask blocks all padding keys), so softmax produces NaN which
        # propagates through the KV-cache and corrupts real positions.
        # Self-attending on zero embeddings: softmax(0)*0 = 0, keeping outputs clean.
        for i in range(N):
            pl = pad_lens[i]
            if pl > 0:
                diag_idx = torch.arange(pl, device=device)
                attention_mask[2 * i, 0, diag_idx, diag_idx] = 0.0
                attention_mask[2 * i + 1, 0, diag_idx, diag_idx] = 0.0

        # 4. Position IDs (per-sequence, accounting for left-padding)
        position_ids = torch.zeros(total_batch, max_len, device=device, dtype=torch.long)
        for i in range(N):
            pl = pad_lens[i]
            rl = real_lens[i]
            pos = torch.arange(rl, device=device)
            position_ids[2 * i, pl:] = pos
            position_ids[2 * i + 1, pl:] = pos

        # 5. Initial forward pass
        hidden_states, past_key_values = self.backbone(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        # 6. Create alignment analyzers (one per sequence)
        analyzers = [None] * N
        if use_alignment_analyzer:
            from .alignment import AlignmentStreamAnalyzer
            for i in range(N):
                pl = pad_lens[i]
                text_start = pl + all_cond_lens[i]
                text_end = text_start + all_text_lens[i]
                analyzers[i] = AlignmentStreamAnalyzer(
                    self.backbone,
                    text_tokens_slice=(text_start, text_end),
                    eos_idx=self.config.stop_speech_token,
                    token_repetition_threshold=token_repetition_threshold,
                    trim_buffer=trim_buffer,
                    batch_idx=2 * i,  # conditioned batch element
                )

        # 7. Precompute padding mask for generation steps (padding positions are fixed)
        max_kv_len = max_len + max_new_tokens + 1
        full_pad_mask = torch.zeros(total_batch, 1, 1, max_kv_len, device=device)
        for i in range(N):
            pl = pad_lens[i]
            if pl > 0:
                full_pad_mask[2 * i, :, :, :pl] = float("-inf")
                full_pad_mask[2 * i + 1, :, :, :pl] = float("-inf")

        # Next position ID per batch element
        next_pos = torch.tensor(
            [real_lens[i // 2] for i in range(total_batch)],
            device=device, dtype=torch.long,
        )

        # Per-sequence generated tokens and state
        generated = [[sos_token.clone()] for _ in range(N)]
        finished = [False] * N

        # 8. Generation loop
        for step in tqdm(range(max_new_tokens), desc="Generating"):
            logits = self.speech_head(hidden_states[:, -1:, :]).squeeze(1)  # (2*N, vocab)

            # Per-sequence: apply CFG, analyzer, process logits, sample
            next_tokens_batch = []
            for i in range(N):
                if finished[i]:
                    tok = torch.tensor(
                        [[self.config.stop_speech_token]], device=device, dtype=torch.long
                    )
                    next_tokens_batch.append(tok)
                    continue

                # CFG
                cond_logits = logits[2 * i: 2 * i + 1]
                uncond_logits = logits[2 * i + 1: 2 * i + 2]
                seq_logits = cond_logits + cfg_weight * (cond_logits - uncond_logits)

                # Alignment analysis
                if analyzers[i] is not None:
                    last_tok = generated[i][-1].item()
                    seq_logits = analyzers[i].step(seq_logits, next_token=last_tok)

                # Process logits
                input_ids = torch.cat(generated[i], dim=1)
                seq_logits = process_logits(
                    seq_logits, input_ids=input_ids,
                    temperature=temperature, top_p=top_p,
                    min_p=min_p, repetition_penalty=repetition_penalty,
                )

                # Sample
                tok = sample_token(seq_logits, do_sample=True)
                generated[i].append(tok)

                if tok.item() == self.config.stop_speech_token:
                    finished[i] = True

                next_tokens_batch.append(tok)

            if all(finished):
                break

            # Build next input embeddings for all batch elements (cond + uncond pairs)
            token_list = []
            for i in range(N):
                tok = next_tokens_batch[i]
                token_list.extend([tok, tok])  # cond and uncond get same token
            next_tokens = torch.cat(token_list, dim=0)  # (2*N, 1)

            next_emb = self.speech_emb(next_tokens)
            if self.speech_pos_emb is not None:
                next_emb = next_emb + self.speech_pos_emb.get_fixed_embedding(step + 1)

            # Position IDs and attention mask for this step
            step_pos = (next_pos + step).unsqueeze(1)  # (2*N, 1)
            kv_len = max_len + step + 1
            step_mask = full_pad_mask[:, :, :, :kv_len]

            hidden_states, past_key_values = self.backbone(
                inputs_embeds=next_emb,
                attention_mask=step_mask,
                position_ids=step_pos,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # 9. Collect results
        results = []
        self._last_batch_forced_eos = []
        self._last_batch_text_complete = []
        for i in range(N):
            tokens = torch.cat(generated[i], dim=1)

            # Trim garbage tail
            if analyzers[i] is not None and analyzers[i].trim_to is not None:
                trim_len = analyzers[i].trim_to + 1
                if trim_len < tokens.shape[1]:
                    logger.info(f"Trimming seq {i} from {tokens.shape[1]} to {trim_len}")
                    tokens = tokens[:, :trim_len]

            results.append(tokens)
            self._last_batch_forced_eos.append(
                analyzers[i].forced_eos if analyzers[i] else False
            )
            self._last_batch_text_complete.append(
                analyzers[i].complete if analyzers[i] else True
            )

        return results

    @torch.inference_mode()
    def inference_turbo(
        self,
        t3_cond: T3Cond,
        text_tokens: Tensor,
        max_gen_len: int = 1000,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
    ) -> Tensor:
        """
        Simplified inference without CFG (for Turbo mode).

        Args:
            t3_cond: Conditioning data
            text_tokens: Text token IDs
            max_gen_len: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty

        Returns:
            Generated speech tokens
        """
        text_tokens = torch.atleast_2d(text_tokens).to(self.device)

        # Start with SOS token
        sos_token = torch.tensor(
            [[self.config.start_speech_token]], device=self.device, dtype=torch.long
        )

        # Prepare initial embeddings (no CFG)
        embeds, _ = self.prepare_input_embeds(
            t3_cond=t3_cond,
            text_tokens=text_tokens,
            speech_tokens=sos_token,
            cfg_weight=0.0,
        )

        # Initial forward
        hidden_states, past_key_values = self.backbone(
            inputs_embeds=embeds,
            use_cache=True,
        )

        # Get first speech logits
        speech_logits = self.speech_head(hidden_states[:, -1:, :])

        # Process and sample
        logits = process_logits(
            speech_logits.squeeze(1),
            input_ids=sos_token,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        next_token = sample_token(logits, do_sample=True)

        generated = [next_token]
        current_token = next_token

        # Generation loop
        for _ in tqdm(range(max_gen_len)):
            # Embed current token
            current_emb = self.speech_emb(current_token)

            # Forward with cache
            hidden_states, past_key_values = self.backbone(
                inputs_embeds=current_emb,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Get logits
            speech_logits = self.speech_head(hidden_states)

            # Process and sample
            input_ids = torch.cat(generated, dim=1)
            logits = process_logits(
                speech_logits.squeeze(1),
                input_ids=input_ids,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )

            # Handle all -inf logits
            if torch.all(logits == float("-inf")):
                print("Warning: All logits are -inf")
                break

            next_token = sample_token(logits, do_sample=True)
            generated.append(next_token)
            current_token = next_token

            # Check for EOS
            if next_token.item() == self.config.stop_speech_token:
                break

        # Concatenate and remove EOS if present
        all_tokens = torch.cat(generated, dim=1)
        if all_tokens.shape[1] > 0 and all_tokens[0, -1] == self.config.stop_speech_token:
            all_tokens = all_tokens[:, :-1]

        return all_tokens
