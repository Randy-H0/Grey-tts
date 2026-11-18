from dataclasses import dataclass, asdict, fields


@dataclass
class Config:
    # codec: DAC 44kHz, 86 frames/s, 9 codebooks x 8-dim latents
    codec: str = "dac"      # "dac" (codebook vectors, snapped) or "kvae" (continuous 64-d latents @ 50 Hz)
    n_q: int = 9            # codebooks to model (fewer = easier, slightly lower fidelity)
    patch: int = 2          # DAC frames per model token (2 -> 43 tokens/s)
    align_q: int = 4        # coarse codebooks used for alignment search (fine ones are ~noise w.r.t. text)

    # text
    add_blank: bool = True  # intersperse blank tokens between characters (helps alignment)

    # text encoder
    enc_dim: int = 256
    enc_layers: int = 4
    enc_heads: int = 4

    # flow-matching decoder
    dec_dim: int = 384
    dec_layers: int = 8
    dec_heads: int = 6

    ffn_mult: int = 4
    dropout: float = 0.1
    n_speakers: int = 1
    use_style: bool = True  # condition on style exaggeration score (0 narration .. 1 acting)
    use_volume: bool = True  # condition on vocal effort (0..0.1 whisper, ~0.5 normal, 1 shouting)
    use_speed: bool = True   # condition on speaking rate (0 slow, 0.5 speaker's normal pace, 1 fast)
    use_mood: bool = False       # sad / negative .. happy / positive (speaker-relative valence)
    use_dominance: bool = False  # meek .. assertive (speaker-relative)
    use_whisper: bool = False    # 0 normal voice .. 1 fully whispered (own control; volume = loudness)
    use_band: bool = False       # recording bandwidth: 1 full band (44.1/48 kHz), 0 = 24 kHz-limited source (JVS); generate at 1
    use_clean: bool = False      # recording cleanliness: 0 hissy / whining .. 1 studio-clean (generate at 1)
    use_guid: bool = False       # guidance as a condition (Smol v14.5, 2026-11-07): distilled from takes made at guidance w,
                                 # g learns w -> one decoder pass per step at any guidance (synthesize / convert route it)
    use_lang: bool = False   # condition on language (en / ja ...); new multilingual runs turn it on
    n_langs: int = 8
    only_speakers: str = ""  # e.g. "a character voice": train on just these speakers (ids stay those of the full data)
    oversample: str = ""   # e.g. "ja:2" -> each Japanese clip appears twice per epoch
    voice_cap: str = ""    # e.g. "kvae_mlsnl:1000" -> at most 1000 clips per voice from that dataset dir (random, seed 0)
    cfg_drop: float = 0.1   # prob. of dropping text conditioning (enables classifier-free guidance)
    label_drop: float = 0.15  # prob. of dropping each of speaker / style / volume / speed independently
    label_drop_all: float = 0.0  # prob. of dropping every label at once (unlabeled use)
    song_cont_prob: float = 0.0  # song clips: prompt = the bars right before them in the same song (melody continuation)
    sing_spk_drop: float = 0.0  # sung clips trained with the singer hidden: singing is learned, not just these voices
    sample_random: int = 0  # >0: that many random lines per language, talked + sung, random voices / tempo / key / genre
    sample_set: str = ""  # json list (configs/sample_set_v12.json): fixed lines per language, talked and sung
    sample_val: bool = True  # False: no style/volume/speed variants of the val lines at sample steps (Grey Studio: only sample_set)
    use_notes: bool = False  # per-frame note input [pitch, sung, onset, known] (data/singing/note_store.pkl)
    note_store: str = "data/singing/note_store.pkl"
    note_drop: float = 0.3  # rows trained without their notes (singing without notes keeps working)
    implicit_align: bool = False  # v13: the decoder gets the text spread evenly over the frames and learns where every
    # syllable goes itself (no duration model / forced alignment in the decoder's input; durations only give the total
    # length of spoken lines). Owner 2026-10-19: the model should place syllables and stretch vowels on its own
    # v16 (you 2026-11-11: "train swift along with the model, remove the bad duration model entirely"): Swift-RM
    # (dacts/rhythm.py, whole-line transformer + g) is the model's ONLY duration model, trained with it from step 0;
    # no FlowDurationPredictor at all. Off: the old head (every model before v16).
    swift_rm: bool = False
    exclude_speakers: str = ""  # comma-separated voice names never trained on (removed voices; dacts/fastdata.py + train.py)
    swift_d: int = 384
    swift_layers: int = 8
    # (you 2026-11-15: Swift-RM put inside v15+ as its duration model): its own learning rate, lr x rhythm_lr_mult -- a
    # trained model's lr (6e-6) is ~15x below what Swift-RM trained at (1e-4); 1.0 = no group of its own
    rhythm_lr_mult: float = 1.0
    # language leak (Solver 11-17, notes_solver/V15PLUS_CONTINUATION.md 5.4 T1): the language tag was dropped in ~23 % of
    # rows and its table rows barely moved (0.02 in the whole v15 line): never drop it, and give the table its own rate
    lang_never_drop: bool = False
    # LoRA fine-tunes (11-17, dacts/lora.py): adapters of this rank on the decoder; only they + the voice table train
    lora_rank: int = 0  # 0 = off (full fine-tune)
    lora_alpha: float = 16.0
    lora_lr_mult: float = 20.0  # adapters can take a much higher rate than full weights (the base can't drift)
    lora_top_blocks: int = 0  # (11-17) adapters only on the top N decoder blocks (0 = all): with the voice rows frozen,
    # the backward pass stops at the lowest adapted block -- faster steps, less memory
    lora_bf16_base: bool = True  # (11-18) the adapted layers' frozen weights in bf16 (same maths under autocast, half the memory)
    lora_train_voice: bool = True  # the new voices' table rows train too (False: they stay at their zero-shot start)
    grad_ckpt: bool = False  # decoder blocks recompute their activations in the backward pass (much less memory, ~25 % slower)
    lang_lr_mult: float = 1.0
    lang_steer: int = 0  # (T2s, Solver 11-17) rank of the voice vector's language subspace (16); 0 = off. The voice's part
    # there always equals the TAG's language mean (training and serving): S0's steer built into the model
    use_dur_span: bool = False  # sung words: the duration model is told each word's / rest's length and note count
    pitch_weight: float = 0.0  # sung note frames: pitch of the one-jump clean estimate vs the real recording's (frozen
    # contour net, semitones, Huber): penalises off-key singing directly (the fm loss weighs pitch like any detail)
    pitch_t_min: float = 0.5  # only rows whose noise level makes the clean estimate meaningful
    sing_vowel_prior: float = 0.0  # sung note frames: alignment bonus for vowel tokens (x per-frame logp spread); 0.8 moves
    # held notes from m/n/consonants onto vowels (MAS split sung words like speech: 23% vowels -> 'MMMM', owner 2026-10-19)
    use_beat: bool = False  # per-frame beat / bar position from the tempo (clips start on a downbeat): a metronome
    use_music: bool = False  # singing labels: sing, bpm, bpb (beats per bar), key, genre (dacts/music.py)
    n_genres: int = 22
    text_hide: float = 0.0  # share of clips trained with their words hidden (one BLANK token): speech from no text
    spk_drop_expressive: float = 0.0  # extra prob. of hiding the speaker on expressive clips (style >= expressive_style),
    expressive_style: float = 0.7     # style label kept: teaches speaker-free acting for style_transfer at inference
    sigma_min: float = 1e-4
    lm_name: str = ""          # e.g. FacebookAI/xlm-roberta-base: text encoder cross-attends to its features
    lm_dim: int = 768
    lm_layers: int = 2
    use_emph: bool = False     # word emphasis control: per-word value 0 (de-emphasised) .. 1 normal .. 2 emphasised
    emph_keep: float = 0.3     # in training, each word shows its measured emphasis this often (else 'unspecified')
    contour_path: str = "configs/contour.pt"  # latents -> pitch / loudness net (train_contour.py)
    use_dlv: bool = False      # delivery encoder: another clip of the same speaker -> how they talk (energy, pace, pitch swings)
    dlv_prob: float = 0.5      # share of training rows that get a delivery reference
    use_prompt: bool = False   # prompted voice cloning: the decoder also sees known (reference) latents + a known-mask
    prompt_prob: float = 0.5   # share of training rows that get part of their own audio as the prompt
    # cross-clip prompting (how cloning is used): this share of rows is "clip(s) A of the same speaker (same session /
    # delivery when known) + clip B"; A (text + audio) is given, only B is learned. A <= cross_max_sec and <= B's length
    cross_prob: float = 0.0
    cross_max_sec: float = 15.0
    # prompt A may be up to this many times B's length (owner 2026-10-29 "squeeze zero shot": prompts of 15-25 s raised SIM
    # +0.08..0.11 at inference, but at 1x training rarely saw one over ~10 s and 20 s+ prompts broke more takes)
    cross_len_mult: float = 1.0
    prompt_t_shift: float = 1.0  # > 1: prompted rows train more at the noisy end (model.flow_loss); 1 = off
    critic_weight: float = 0.0  # voice critic in the loop (dacts/voice_critic.py): the model clones, WavLM-SV judges; 0 = off
    critic_every: int = 4       # every Nth step (a rollout costs about one training step: launch-bound)
    # 2026-11-13 (you: "do whatever you can to boost performance"; Solver's idea): a frozen older decoder (v15 XL) as a
    # teacher next to the real target: the same x_t / t / conditioning / prompt go through both, and the student's
    # velocity is also pulled toward the teacher's on the generated frames. A fresh decoder (v16) gets the lineage's
    # 3M steps of cloning as a steady target. 0 = off; the v16 gate turns it off once SIM reaches 0.60.
    teacher_ckpt: str = ""
    teacher_weight: float = 0.0
    critic_rows: int = 4        # prompted rows per critic step (batched: free when launch-bound)
    critic_group: int = 1       # rows fingerprinted per call (2026-11-07): 1 = one at a time (least memory), 4 = all at once
    critic_bf16: bool = False   # (2026-11-09) critic decode + fingerprint in bf16 (faster; gradients ~5% different in direction)
    critic_steps: int = 4       # Euler steps of the rollout (sway-scheduled, as the site samples)
    critic_win: int = 300       # window of a row the rollout runs on (decoder steps): prompt end + start of the take
    critic_fast: bool = False   # compiled decoder, no checkpointing (more memory, no recomputation)
    critic_compile: str = ""    # "" eager | "default" (compiled rollout step + KVAE decode) | "graphs" (+ CUDA graphs): same maths
    critic_sv: str = "microsoft/wavlm-base-plus-sv"  # (11-14, you "max 1 critic") the critic's verifier; never the judge's model
    critic_asr: str = ""          # (11-14, you "we can add another critic") word critic: a frozen CTC recognizer, never the gate's Whisper
    critic_asr_weight: float = 0.0  # its KL(real clip letters || take letters) weight; 0 = off (needs critic_group > 1)
    critic_contrast_weight: float = 0.0  # (11-14, you "do sound not like other voices") InfoNCE vs a bank of earlier prompts; 0 = off (grouped path)
    critic_contrast_m: float = 0.2      # its margin (own must beat the nearest other voice by this)
    critic_contrast_tau: float = 0.05   # its temperature
    critic_k_min: int = 100             # (11-14 Solver) frames of prompt the critic window must hold (150 = 3 s: the 3 s target always fires)
    critic_eval: bool = False           # (11-14 Solver) decoder in eval mode (no dropout) for the critic's rollout + its backward
    critic_skip_alt: bool = False       # (11-14 Solver) leave rows with a damaged / low-passed TARGET out of the critic
    critic_skip_up: bool = False        # ... and upsampled targets too (Solver 11-14: they keep the voice, CAM++ 0.94-0.99 to the original)
    critic_sv2: str = ""                # (11-14, you via Solver) second verifier next to critic_sv: "eres2" = ERes2NetV2 zh (dacts/eres2.py)
    critic_sv_weight: float = 1.0       # inside the voice term: w1 * (1 - cos sv) + w2 * (1 - cos sv2)  (Solver: 0.7 / 0.35)
    critic_sv2_weight: float = 0.0
    critic_contrast_bank: int = 256     # earlier prompts kept as negatives (more = harder look-alikes; Solver 11-14: 1024)
    critic_zs: bool = False     # critic only on zero-shot rows (cross prompt + no learned table vector): what the site does;
                                # v14xl-vtv 0-85k judged any prompted row and its own metric rose 0.75->0.79 while the
                                # outside verifier (Seed-TTS) stayed flat (2026-10-31)
    use_vtv: bool = False       # voice-to-voice rows (dacts/vtv.py): content + contour instead of text; TTS stays as is
    vtv_rows: int = 4           # rows per step that get the recording's content instead of the text
    vtv_shift: float = 4.0      # +- semitones of pitch + formant shift before ContentVec (keeps the source voice out)
    # "exact pitch" mode (you 2026-10-31: "just teach it to handle sticking to pitch well", "like exact pitch", "singing
    # mode does it exactly, it doesnt have to be exact for speech"): a share of the VTV rows is flagged (DacTTS.vtv_exact,
    # convert(exact_pitch=True)); their contour's pitch is RMVPE's (as singing mode gives it), their content comes from
    # audio bent by a random pitch curve, a new target every ~vtv_exact_seg s (the source's own melody becomes useless:
    # only the contour says where the pitch goes), and they get pitch-following losses. Unflagged rows: as before
    vtv_exact_prob: float = 0.0   # share of VTV rows in exact-pitch mode; 0 = off (and no vtv_exact parameter)
    vtv_exact_shift: float = 7.0  # +- semitones of that curve (pitch + formants, as vtv_shift)
    vtv_exact_seg: float = 0.6    # mean seconds between its knots (each gap 0.5-1.5x that; straight glides between)
    vtv_pitch_weight: float = 0.0 # pitch-following loss on those rows (ContourNet on the one-jump estimate vs target)
    vtv_rmvpe_weight: float = 0.0 # ... and judged by RMVPE on decoded 3 s crops (dacts.vtv.PitchLoss: cents + salience)
    vtv_rmvpe_t_min: float = 0.6  # only rows this clean (t >= this): the decoded estimate must have a pitch to read
    # voice-to-voice-only voices (you 2026-11-02, solo singing for VTV singing mode: "make sure singing also sounds good"):
    # rows of voices named with this prefix (tools/vtv_singing_clips.py: sing_*) are ALWAYS VTV rows, on top of vtv_rows,
    # so they never teach text (they have none, notext) or blank speech to sing. "" = off
    vtv_only_prefix: str = ""
    vtv_only_exact: float = -1.0  # their exact-pitch share; < 0 = vtv_exact_prob
    # plan the cross-clip prompts in the sampler (same rules / odds), so batches are packed by each row's REAL length
    # instead of the worst case (2 x length + 3 s): measured 2026-10-21, rows used 53% of that budget, batches 71%
    plan_cross: bool = False
    # prompt labels (owner 2026-10-22): the model is told how its prompt was delivered (whisper, style, mood,
    # dominance, volume, speed, clean, band of the given reference audio), each optional -- so "this reference is a
    # whisper / acted / noisy" is an input instead of something to pick references around. Zero-init (no effect at first)
    use_prompt_labels: bool = False
    prompt_label_drop_all: float = 0.3  # all prompt labels hidden at once this often (references nobody labelled)
    # whispered prompts (with use_prompt_labels): whisper rows continue the same speaker's whispers this often (else
    # normal speech, as before); other rows of a speaker who has whispers continue a whisper this often (-> "the
    # reference whispers, but this line is normal" is learned too)
    cross_whisper_prob: float = 0.0
    cross_whisper_any: float = 0.0
    # per-clip reference labels (owner 2026-10-22): every reference frame / word carries the labels of ITS clip, plus
    # the difference to the labels asked for the new line, so several differently delivered clips can be given at once
    use_clip_labels: bool = False
    # mixed references (use_clip_labels): this share of cross rows takes its reference clips from anywhere in the
    # speaker's recordings (other sessions / deliveries, whispers too) instead of the same session -- so the model
    # practises taking the voice from all clips and the delivery from the one that matches the new line
    cross_mix_prob: float = 0.0
    longest_first: bool = False  # memory tests only: the longest (most memory-hungry) batches come first
    fast_data: bool = False  # data index from a prebuilt cache (dacts/fastdata.py): starts in seconds, workers share it
    use_pctx: bool = False  # prompt prosody: the text encoder sees each known token's length / pitch / loudness
    # a clip's damaged copy (make_corrupt.py) as the TARGET, labelled dirty (clean = 0), this often: the same voices
    # then appear both clean and noisy, so hiss / noise is learned as the clean label's job, not part of a voice
    dirty_target_prob: float = 0.0
    corrupt_prob: float = 0.0  # prompted rows with a damaged copy (make_corrupt.py): show it as the prompt this often
    prompt_spk_drop: float = 0.8  # on those rows, hide the speaker label this often (must listen to the prompt)
    zs_weight: float = 0.0     # zero-shot voice encoder (x-vector -> speaker vector); 0 = off. Weight of its match-the-table loss
    zs_prob: float = 0.5       # share of training rows that use the encoder's vector instead of the speaker table
    zs_in: int = 512           # x-vector size (WavLM-base-plus-sv)
    zs_max_refs: int = 8       # references averaged per row: 1..this many other clips of the same speaker
    upsampled_prob: float = 0.0  # share of rows (that have one) whose target is the AudioSR 48 kHz copy (latents_upsampled)
    upsampled_band: float = 0.9   # band label of those copies: near full band, distinguishable from real full band
    clean_band_rule: bool = False  # clean label *= band label: muffled recordings count as (partly) dirty
    lowpass_prob: float = 0.0  # share of rows whose target is a low-passed copy (make_corrupt.py --kind lowpass), band-labelled
    use_vre: bool = False      # voice reference encoder: other clips' latents -> voice vector (reads the sound, not an x-vector)
    vre_prob: float = 0.5      # share of rows whose voice comes from it (after vre_warmup)
    vre_warmup: int = 3000     # steps where it only learns to match the speaker table (its output unused)
    vre_weight: float = 1.0    # weight of that match loss
    vre_max_sec: float = 20.0  # reference length (1-4 other clips, concatenated)
    # v14 reference segment (VoxCPM2 / MiniMax-Speech style): 1-4 OTHER clips of the speaker (3-vre_max_sec s, the
    # use_vre sampling), read frame by frame by a small encoder, and the decoder cross-attends to them -- the voice
    # from its sound, next to the continuation prompt, not squeezed into one vector (that alone gave SIM 0.19)
    use_ref_seg: bool = False
    ref_seg_prob: float = 0.8   # share of rows that get it (the rest learn without: reference guidance, no-reference use)
    ref_seg_every: int = 4      # a cross-attention layer after every this many decoder blocks
    ref_x_all: bool = False     # (2026-11-13, v16 XL: 2 look-ups vs v15's 11) + one after EVERY other decoder block too
                                # (decoder.ref_x2, zero-init: the trained ref_x keep their places, the model is unchanged)
    ref_seg_dim: int = 384      # reference encoder width
    ref_seg_layers: int = 4     # reference encoder blocks
    ref_seg_tokens: int = 0     # >0: squeeze the reference into this many voice tokens (owner 2026-10-28: fast); 0 = every frame
    sim_weight: float = 0.0    # speaker-similarity loss (dacts/sim_loss.py): decoded prediction vs real crop, WavLM-SV
    sim_t_min: float = 0.6     # only rows this clean (t >= this): earlier, the one-jump estimate is not a voice yet
    sim_rows: int = 2          # rows (3 s crops) per step that get it
    zs_noise: float = 0.0      # jitter the reference fingerprint (noise norm up to this share of its own norm):
                               # a smoother fingerprint -> voice mapping for voices that fall between training speakers
    zs_holdout: str = ""       # speakers the encoder never hears (honest zero-shot evaluation)
    zs_sample_refs: str = ""   # "name=ref.wav,...": training samples also clone these voices zero-shot
    zs_sample_texts: str = ""  # '|'-separated (or @file) lines those cloned voices read
    repa_until: int = 0          # stop the REPA loss at this step (0 = never); its features are then not even loaded
    repa_weight: float = 0.0   # REPA: align a decoder layer with speech-SSL features of the real audio (0 = off)
    repa_layer: int = 2        # decoder block whose output is aligned (of dec_layers; ~1/3 deep as in the paper)
    repa_dim: int = 256        # target dims (PCA of XLS-R features, extract_repa.py)
    spk_repa_weight: float = 0.0  # Speaker-REPA (research idea 1, 2026-10-28): middle decoder layers, pooled over the generated
                                  # frames, -> cosine to the voice's x-vector (other clips; needs zs_weight > 0 for the targets)
    spk_repa_layers: str = ""      # decoder blocks, e.g. "7,14,22" ("" = dec_layers // 6, // 3, // 2)
    ctx_drop: float = 0.3      # prob. of hiding the previous line's text in training (works with and without)
    code_head: bool = False   # code picker: samples the DAC codes codebook by codebook (dacts/codehead.py)
    code_lr: float = 3e-4     # its own optimizer; it never changes the flow model
    code_t: float = 0.5       # sampling hands over to the picker at this noise level
    code_t_range: tuple = (0.4, 0.7)  # training range around it

    # training
    lr: float = 3e-4
    # U-Net long skips in the decoder (U-ViT): block i of the second half also gets the output of its mirror block in
    # the first half, through a low-rank projection whose up-weights start at zero (no effect until learned)
    dec_skips: bool = False
    dec_skip_rank: int = 256
    # parameters added by growing a model (comma-separated name prefixes, set by the grow script) learn at
    # lr x new_lr_mult: they start as identities with everything to learn, the old ones only need fine-tuning
    new_params: str = ""
    new_lr_mult: float = 1.0
    spk_lr_mult: float = 1.0  # speaker table's own learning rate (x lr): its rows only get sparse updates
    warmup: int = 1000
    lr_restart: int = 0           # step of a warm restart (0 = none): re-warm to lr_restart_peak * lr
    lr_restart_peak: float = 0.5
    lr_drop_at: int = 0           # from this step on, the whole schedule x lr_drop (2026-11-07: "continue ... with a lower
    lr_drop: float = 1.0          # learning rate"; a factor in the schedule, so resumes never compound it)
    max_steps: int = 60_000
    batch_patches: int = 12_000   # total (padded) decoder tokens per batch
    max_seconds: float = 20.0
    ema: float = 0.999
    grad_clip: float = 1.0
    num_workers: int = 2
    compile: bool = True    # torch.compile encoder/decoder (Triton kernels via Inductor)
    flex_attn: bool = False  # training: the decoder's self-attention via FlexAttention (Triton; faster than SDPA's
                             # memory-efficient kernel on Windows, same maths); inference is unaffected
    max_vram_gb: float = 0  # cap this process' GPU memory (0 = no cap); cached blocks are released above it
    log_every: int = 50
    sample_every: int = 2_000
    ckpt_every: int = 2_000
    n_val: int = 8
    # validation clips (sample sentences from real clips): "auto" = hold out n_val, but with under 20 clips also train
    # on them; "10%" = hold out that share; "both" = n_val clips for samples that are ALSO trained on (tiny data)
    val_split: str = "auto"
    keep_ckpts: bool = False  # also keep a small inference snapshot (EMA, fp16) at every checkpoint
    keep_every: int = 0  # ... only at multiples of this step (0 = every checkpoint; v14xl-vtv: saves every 5k for the
    # dashboard's VTV samples, snapshots kept every 20k)
    sample_texts: str = ""  # extra '|'-separated texts for training samples (unseen lines, first val speaker)

    @property
    def latent_dim(self):
        return 64 if self.codec == "kvae" else self.n_q * 8

    @property
    def token_dim(self):
        return self.latent_dim * self.patch

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def override(self, pairs):
        """Apply 'key=value' overrides from the command line."""
        types = {f.name: f.type for f in fields(self)}
        for p in pairs or []:
            k, v = p.split("=", 1)
            if k not in types:
                raise KeyError(f"unknown config key: {k}")
            t = types[k]
            t = {"int": int, "float": float, "bool": bool}.get(t, t) if isinstance(t, str) else t
            setattr(self, k, v.lower() in ("1", "true", "yes") if t is bool else t(v))
        return self
