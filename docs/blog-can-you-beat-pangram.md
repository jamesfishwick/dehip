## Could you just fine-tune your way past it?

Short answer: yes. Longer answer: yes, but there are two versions of "yes," and they're not close.

**Beat this Pangram.** A detector is a fixed function. If you can see its output, you can train against it. Take Pangram's human-score, make it the reward, and train the paraphraser to push that number up while keeping the meaning intact. Textbook adversarial ML, and it works. You'd clear the bar. The catch is what "the bar" means: you've overfit to one snapshot of one detector. Pangram releases a new version, your model goes back to failing, and you're on a treadmill you're paying to run.

**Beat detection itself.** That one's worth thinking about. A 2023 result from Sadasivan and co. says the best any detector can ever do is capped by how far your text sits from the way humans actually write. Push your output's distribution toward the human one and every detector, present and future, slides toward a coin flip. Undetectable in the limit. You get there by writing like a person, and that is hard.

Then comes the tradeoff you can't dodge. You want two things at once: say a specific thing (a faithful rewrite of this draft) and look distributionally human. Pin the content down and you lose the room you'd need to roam the whole human distribution. That leftover is exactly what a detector reads: the fingerprint of text pushed toward a goal, which a person's writing does not carry.

Which is where HIP lands. It never trained against Pangram. It trained on a stand-in goal, sound like human prose, and hoped that carried over. Pangram got to learn HIP's own fingerprint instead. Training against Pangram directly would erase its specific signal and probably clear the bar, but that's the first kind of "yes," the treadmill one.

And the treadmill is expensive. My whole test ran about two dollars: a handful of API calls and forty minutes of a rented GPU. Beating Pangram by fine-tuning is a different animal. Tens of thousands of queries to clone its judgment, then reinforcement learning (the expensive kind of GPU time, not the cheap inference kind), then iterate. Weeks and low thousands of dollars, for a result that dies the next time Pangram retrains. One training run on their side wipes out everyone's evasion work at once. That asymmetry is the business model.

HIP went with the cheap lever. Which I found out for two dollars instead of two thousand.
