# A 0.6B model beat a 14B model at looking human, and wrote worse

Draft. My voice, needs a polish pass and a Pangram check before it ships.

---

There is a paper called "Base Models Look Human To AI Detectors." The finding underneath it is strange and a little funny. A base language model, the raw thing before anyone teaches it to follow instructions, scores as almost perfectly human on AI detectors like Pangram and GPTZero. Around 97 to 99 percent human. Teach that same model to follow instructions, and its writing drops to something like 17 to 30 percent human. Instruction tuning is what makes a model sound like a model.

The authors ship a fix for that. It is called HIP, humanization by iterative paraphrasing. You take a base model, train a small adapter on pairs of AI-written and human-written text, and use it to rewrite instruction-model output back toward the human distribution. They released the adapters. So I built the thing that measures whether it works, wired up the cascade end to end, and ran it.

I want to be careful about what "works" means here, because that is the whole point.

## Two kinds of ruler

The eval has three metrics. Two of them measure distribution. MMD compares text embeddings, and token-L2 compares word-frequency vectors. Both answer a version of the question "does this look statistically like human writing." The third metric is a judge. It reads two pieces and picks which is better on clarity, coherence, creativity, depth, and relevance. That one answers a different question. "Is this actually good writing."

I ran a small smoke test. Fifty prompts, drafts from a 4B instruction model, rewrites from the smallest released HIP adapter, which is 0.6B. This is the weak model. I ran it first to prove the pipeline, not to get a good result. Then I scored the drafts and the rewrites against the human references on all three metrics, and I ran Pangram on both sets.

Here is what came back.

The rewrite moved every distribution metric toward human. MMD dropped by a factor of nine. Token-L2 dropped. Pangram rated the rewrites 27 points more human than the drafts, and flipped a quarter of them from fully AI to fully human. On the "looks human" question, the tiny model did real work.

The judge went the other way. Every dimension got worse. Clarity, coherence, creativity, depth, relevance, all down. The rewrites read as more human and as worse writing at the same time.

## The part that made me laugh

The 0.6B rewrite scored a lower MMD than the published number for the 14B version of the proprietary system this is all chasing. A tiny model, prone to repeating itself and mangling length, beat a 14B production model on the distribution metric.

It did that by making the text blander. Human writing is less polished than instruction-model output, so if you sand off the polish you move toward the human distribution, and the detector relaxes. You have not written anything better. You have written something more forgettable, and the metric that only knows about distribution cannot tell the difference.

If I had shipped a detector-only evaluation, or an MMD-only one, this run reads as a triumph. Because I measured quality separately, it reads as what it is. The humanizer is a humanizer. It is not a writer.

## So can you beat Pangram

Yes, and that turns out to be the boring part. The released adapters were trained to do exactly that, and even the smallest one gets a quarter of the way there on a bad day. The 4B will almost certainly clear the bar.

The question worth asking is the one nobody has answered. Can a stronger paraphraser pull the text toward human and past the detector without wrecking the prose. The 0.6B cannot. Whether the 4B can is the experiment I am running next, and now I have an instrument that will tell me the truth either way, because it refuses to let a low detector score stand in for good writing.

That refusal is the entire value of building the eval instead of trusting one number.
