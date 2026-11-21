# HolyZip

**Question:** how well would a large pretrained language model, driving an arithmetic coder, compress the Bible in many languages? The idea was prompted by [tufazip](https://tufalabs.ai/research/tufazip/) and the model in mind is Qwen3.6-35B-A3B.

This repository holds a first, CPU-only pilot that puts numbers on that question. It is a measurement harness, not a finished compressor. The 35B model does not fit the pilot machine (4 cores, 15 GB RAM, no GPU), so two small members of the same model family stand in for it: `Qwen3.5-0.8B-Base` and `Qwen3.5-2B-Base`. Nothing in here was measured with the 35B model.

All tables are in [pilot/RESULTS.md](pilot/RESULTS.md).

## What the pilot found

Sizes are in bits per byte of the original file; lower is better. "PPMd" is the best of the five classical compressors that were run (xz, bzip2, zstd, brotli, PPMd).

**1. For languages the model knows, even the 0.8B model beats PPMd clearly.**

| translation | 0.8B model | PPMd | model as share of PPMd (95% interval) |
|---|--:|--:|--:|
| King James Version | 0.75 | 1.46 | 51% (47–56) |
| Louis Segond 1910 (French) | 0.84 | 1.43 | 59% (53–65) |
| Russian Synodal | 0.71 | 1.13 | 63% (58–68) |
| Chinese Union Version | 1.16 | 1.76 | 66% (61–70) |
| Luther 1912 | 1.07 | 1.55 | 69% (64–75) |
| Byzantine Greek NT | 0.84 | 1.06 | 79% (73–85) |
| Hindi IRV | 0.61 | 0.72 | 84% (78–90) |

**2. For languages it knows poorly, the advantage is gone.** Swahili comes out at 102% of PPMd (93–111), which is a tie. Pointed Biblical Hebrew is a tie too: 107% (94–121) on the original bytes, 100% (86–113) when the text is scored in the normalised form the model was trained on. Miniafia, a language of Papua New Guinea, comes out at 140% (133–148): the model is clearly worse than PPMd, and the 2B model barely improves on that (134%). Most of the 1,024 languages with a redistributable translation on eBible.org are small languages like Miniafia.

**3. On the famous English text the larger model is recalling, not predicting.** Going from 0.8B to 2B cuts the King James cost from 0.75 to 0.50 bits per byte, which is 34% of PPMd (29–39). The gain sits where the text is best known: the six Gospel and Acts chapters drop from 0.65 to 0.26, Matthew 2 to 0.06, and four famous chapters (Genesis 1, Psalm 23, John 1, 1 Corinthians 13) to 0.07 each. Published results for language models on text they have not seen are 0.5–0.75 bits per character even for models of 7B to 70B parameters, so numbers this low are recall. Luther 1912 improves much less (1.07 to 0.93, 60% of PPMd). A 35B model has presumably memorised far more, so its King James result would mostly measure memorisation.

**4. Putting the same verse of an already-decoded translation into the context is the strongest lever, but only between languages the model knows.**

| target | translation in context | alone | with it | change |
|---|---|--:|--:|--:|
| World English Bible | King James | 0.64 | 0.27 | −57% |
| Elberfelder | Luther 1912 | 0.98 | 0.50 | −49% |
| Luther 1912 | King James | 1.05 | 0.65 | −39% |
| Chinese Union | King James | 1.08 | 0.71 | −34% |
| Swahili | King James | 1.79 | 1.67 | −7% |
| Miniafia | King James | 2.10 | 2.12 | +1% |

(0.8B model, six New Testament chapters.) The 2B model shows the same pattern: Luther 1912 given the King James drops by 41% (0.86 to 0.51), Miniafia does not move (+2%). A classical compressor given the same help gains far less: with both translations in one stream, xz and PPMd save 12–23% on the second translation within a language and nothing across languages.

**5. The pre-trained "Base" checkpoint predicts raw text better than the chat-tuned one.** At 0.8B the chat-tuned model needs 25% more bits on the King James chapters and 8% more on Luther 1912. Qwen3.6-35B-A3B exists only as a chat-tuned model; its closest pre-trained sibling is `Qwen3.5-35B-A3B-Base`.

**6. A real lossless round trip works, and it is fragile.** One chapter (Joshua 16, Luther 1912, 1,425 bytes) was compressed to 249 bytes with a real arithmetic coder and decoded back to identical bytes. But the decoder only works if it reproduces the encoder's probabilities bit for bit:

- Running the model in one batched pass instead of token by token changes the compressed size by nothing measurable (244.8 bytes either way) yet makes the decoder fail at the second token.
- A reviewer's emulation of a different CPU instruction set (AVX2 instead of AVX-512) broke decoding from the third token.
- The same code path run twice, run in separate processes, and run with 1 versus 2 threads gave identical probability tables on this machine.

**7. The tokenizer is not lossless out of the box.** It normalises text to Unicode NFC, which silently rewrites the pointed Hebrew of the Westminster Leningrad Codex. The pilot switches the normaliser off (`pilot/common.py`).

## What this means for the idea

Background facts in this section come from a separate literature check; the sources are linked.

- **It is not the tufazip principle.** tufazip trains a 163M-parameter model while it compresses, and the benchmark it competes in counts the decoder and the 4.3 MB of weights it ships ([write-up](https://tufalabs.ai/research/tufazip/), [benchmark rules](https://www.mattmahoney.net/dc/textrules.html)). Using a pretrained 35B model treats the model as a shared prior that is not counted, as [LLMZip](https://arxiv.org/abs/2306.04050), [ts_zip](https://bellard.org/ts_zip/) and [llama-zip](https://github.com/AlexBuz/llama-zip) do. Counted, the model is 72 GB at 16 bit and about 22 GB at 4 bit; all redistributable eBible text together is roughly 2.4–2.8 GB.
- **The single-language case is prior art.** llama-zip already lists `Qwen3.5-35B-A3B-Base` as a predictor (about 0.93 bits per byte on the Calgary novel `book1`), and an 8B Llama does as well on prose in the same table. No published work was found that compresses the Bible with a pretrained language model, or that codes one translation given another with one.
- **Model choice.** For raw-text prediction `Qwen3.5-35B-A3B-Base` is the better fit than the chat-tuned Qwen3.6-35B-A3B (finding 5); there is no Base release of 3.6.
- **Determinism is the main engineering risk** (finding 6). vLLM's batch-invariant mode [refuses](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/selector.py) this architecture's linear-attention layers, and llama-zip warns that its output may not decode on another system.
- **What looks worth doing.** Not "the smallest Bible", which a large model wins by having memorised it, but two measurements nobody seems to have published: bits per verse by language and translation as a map of what the model knows, and how much each further translation costs once others are decoded (finding 4).

## How it was measured

- **Texts.** 17 translations from [eBible.org](https://ebible.org/Scriptures/) in its verse-per-line format (`GEN 1:1 In the beginning ...`). The file as downloaded is the thing to compress, verse references included. Texts are fetched on demand and not committed; all 17 are marked redistributable by eBible, most are public domain.
- **Sample.** 12 chapters drawn with a fixed seed, 6 from the Old and 6 from the New Testament: Leviticus 2, Joshua 16, 2 Kings 11, 2 Chronicles 30, Isaiah 13, Jeremiah 31, Matthew 2 and 20, Luke 7 and 16, Acts 22 and 28. Hebrew has only the 6 Old Testament chapters, Greek and Miniafia only the 6 New Testament ones.
- **Model cost.** For each chapter, the sum of −log2 p(token) over its tokens, with the 12 preceding lines of the same translation as context. This is the size an arithmetic coder driven by the model would emit, to within a few bytes. The model weights are not counted.
- **Whole-file estimate.** Old and New Testament chapters are weighted by their share of the file. The interval treats chapters as the sampling unit.
- **Classical compressors.** Run on the whole file, which is their best case.
- **Translation in context.** Before each verse of the target, the same verse of the other translation is inserted; only the target's tokens are charged.
- **Round trip.** `pilot/ac.py` is an exact integer arithmetic coder; `pilot/ac_roundtrip.py` compresses and decompresses one chapter token by token and compares probability tables between code paths.

The measurement code was reviewed by independent agents, which re-derived stored records, unit-tested the coder and found no error in the numbers. The limits below come largely from that review.

## Limits

- **Small, Gospel-heavy sample.** All six New Testament chapters happen to be from the Gospels and Acts; there is no poetry and no epistle. The intervals describe only the kinds of text that were drawn.
- **Verse references favour the model.** PPMd spends about 17 bits per verse on the `GEN 1:1` prefixes, the model well under 1. On scripture text alone the shares in finding 1 would be about 4–7 points higher (King James about 55% instead of 51%).
- **Stronger classical compressors were not run.** cmix and paq8px are expected to be 10–30% smaller than PPMd; that figure is taken from other corpora and was not measured here. Against such a baseline the 0.8B model would stay clearly ahead only for English.
- **The weights are not counted.** The 0.8B model is 1.7 GB and Qwen3.6-35B-A3B is 72 GB at 16 bit, against 79 MB for all 17 files together. Under the rules tufazip competes under, where the decompressor and its weights count, none of this is a gain.
- **Bits per byte are not comparable across scripts.** Compare within a row, not between rows.
- **Speed.** The costs come from batched passes. A real codec must run token by token in both directions; this pilot code manages 3.6 tokens per second for the 0.8B model on one busy CPU thread.
- **Determinism was only shown on one machine.**
- **Translation in context:** the inserted verse is looked up by the target's own verse number, which tells the model that number (at most 1% of the cost), and the "alone" figure has 12 lines of own-language context against 6 in the other column.

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers tokenizers safetensors huggingface_hub numpy zstandard brotli pyppmd
python pilot/fetch_bibles.py       # downloads the translations to data/vpl/
python pilot/make_sample.py        # writes pilot/results/sample.json
python pilot/baselines.py          # classical compressors, about 6 minutes
./pilot/run_queue.sh               # model measurements, about 1.5 hours on 4 cores
python pilot/ac_roundtrip.py --book JOS --chapter 16 --threads 1 --alt-threads 2
python pilot/report.py > pilot/RESULTS.md
```

eBible regenerates its files from time to time, so byte counts can drift from the ones recorded here.
