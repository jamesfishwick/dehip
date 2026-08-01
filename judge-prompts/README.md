# JMQ judge prompts

Transcribed verbatim from Appendix 9 of the Rosmine DFT post (the PDF is the only source). The post shows the PROMPT / CANDIDATE A / CANDIDATE B template once and elides it in the later prompts; here it is included in every file so each is directly usable. The "Given the a prompt" typo in `overall.txt` is in the original.

Usage per the post: present prompt plus human and model completions in randomized A/B order to prevent positional bias. JMQ = 2x model win-rate, so 1.0 (a 50% win rate) is optimal when the goal is matching human text. Judge in the post was GPT-5.4-mini.

Files: `overall.txt`, `coherence.txt`, `creativity.txt`, `clarity.txt`, `relevance.txt`, `depth.txt`.
