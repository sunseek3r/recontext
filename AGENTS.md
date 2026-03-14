It is a Hackathon PoC Project. Here is a short version of task statements:

# Statements
- Objective: Build a context collection pipeline to retrieve relevant code snippets for code completion tasks.

- Evaluation Metric: ChrF score (character n-gram F-score) comparing model completions against ground truth.

- Target Models & Context Windows:
	- Mellum (JetBrains): 8K tokens.
	- Codestral (Mistral AI): 16K tokens.
	- Qwen2.5-Coder (Alibaba Cloud): 16K tokens.

- Formatting Requirements:
	- Use the special token <|file_sep|> to separate different files/snippets within your context.
	- Trimming Logic: The evaluator trims context from the left. Place the most important/relevant information at the end of your context string.
	- Final Output: Must be in JSONL format.

- Task Type: Fill-in-the-middle (FIM). The system provides the prefix and suffix; your job is to provide the "context" that helps the model predict the missing code.

# Structure
- `data` is a symlink to `EnsembleAI2026-starter-kit/data` dir. It contains repositories with code to find snippets it
- 

# Examples
- Here is an example of JSONL (formatted for readability) input that we are going to use for getting relevant context
```json
{
    "id": "3c4689", 
    "repo": "celery/kombu", 
    "revision": "0d3b1e254f9178828f62b7b84f0307882e28e2a0",
    "path": "t/integration/test_redis.py", 
    "modified": ["t/integration/common.py", "t/integration/test_redis.py"], 
    "prefix": "<FILE_PREFIX_SNIPPET>",
    "suffix": "<FILE_SUFFIX_SNIPPET>", 
    "archive": "celery__kombu-0d3b1e254f9178828f62b7b84f0307882e28e2a0.zip"
}
```
- Here is an example of prepared context:
```json
{"context": "<|file_sep|>t/unit/utils/test_uuid.py\nfrom __future__ import absolute_import, unicode_literals\n\nfrom kombu.utils.uuid import uuid\n\n\nclass test_UUID:\n\n    def test_uuid4(self):\n        assert uuid() != uuid()\n\n    def test_uuid(self):\n        i1 = uuid()\n        i2 = uuid()\n        assert isinstance(i1, str)\n        assert i1 != i2\n"}
```

# Implementation
We are currently working on implementation that consists of several modules:
