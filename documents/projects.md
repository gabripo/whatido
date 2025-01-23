# Projects

## Tenders and clause-by-clause
- Based on Gemini AI project (demo already there)
- Add creation of documents (specifications)
- For software companies, generate code out of specifications

## Video analysis for productivity increase
### Aspects
- Advantages over CoPilot on MS Office:
	- Identify skills of each employee, assigning useful tasks based on the skills to improve - the manager chooses them among a pool of options, provided by the AI.
	- Use the entire PC, not signle applications (like MS Office).
### INPUT
- Recorded activity from the employee (video, e-mails, audio...)
- Analyze videos to extract some metrics to define productivity.
- Best practices defined by web researches (agentzzzz!) like papers, tutorials, ....
### TRAINING
- Fine-tune a vision model (llama?) based on the best practices (label: "best practice of X" with economics metrics, video: its video)
- Each "best practice of X" has its own voting system over the metrics (Example: "write an e-mail to one colleague to ask for information" has different metrics for English fluency about writing a clear e-mail, technical knowledge about using Office)
- AutoML when asked: learn from what the (skilled) user does.
### OUTPUT
- Provide recommendations to the employee to increase his or her productivity (based on PC use and to economic models of productivity)
- Record the employee's usage data to build a soft/hard skill profile based on which the manager can assign tasks.
	- Clustering skills: writing code, Excel usage, English fluency.
	- Recognize people's actual behaviour and personality traits within a meeting / e-mail / calls (sentiment analysis?).
### DEMO IDEA
- Fine-tune of llama vision:
	- Training data: screenshot with an e-mail, metrics about the e-mail (the best technical e-mail 10/10, best arrogant e-mail 10/10, worst clear e-mail 1/10), use dataset augmentation (blur, crop, rotate ...) to enlarge the dataset, use LLM to generate e-mails?
	- Input: average employee writes an e-mail.
	- Output: tips on how to improve the written e-mail, profile of the employee (personality and skills).
- IF ENOUGH TIME: front-end showing the metrics (in real-time?).
