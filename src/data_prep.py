import json
from pathlib import Path


def build_claim(prompt: str, subject: str, target: str) -> str:
	base = prompt.format(subject) if "{}" in prompt else f"{subject} {prompt}".strip()
	separator = "" if base.endswith((" ", "\n", "\t")) else " "
	return f"{base}{separator}{target}".strip()


def convert_counterfact_to_jsonl(input_path: Path, output_path: Path) -> int:
	with input_path.open("r", encoding="utf-8") as f:
		records = json.load(f)

	written = 0
	with output_path.open("w", encoding="utf-8") as out:
		for item in records:
			if not item or "requested_rewrite" not in item:
				continue

			rewrite = item["requested_rewrite"]
			case_id = item.get("case_id")
			subject = rewrite.get("subject", "")
			prompt = rewrite.get("prompt", "")
			target_true = rewrite.get("target_true", {}).get("str", "")
			target_new = rewrite.get("target_new", {}).get("str", "")

			if case_id is None or not prompt or not subject or not target_true or not target_new:
				continue

			row = {
				"case_id": case_id,
				"correct": build_claim(prompt, subject, target_true),
				"incorrect": build_claim(prompt, subject, target_new),
			}
			out.write(json.dumps(row, ensure_ascii=False) + "\n")
			written += 1

	return written


def main() -> None:
	data_dir = Path(__file__).resolve().parent.parent / "data"
	input_path = data_dir / "counterfact.json"
	output_path = data_dir / "counterfact.jsonl"

	written = convert_counterfact_to_jsonl(input_path, output_path)
	print(f"Wrote {written} rows to {output_path}")


if __name__ == "__main__":
	main()