import argparse
import json
import math
import sys


def count_tokens(text, token_size=4):
    """Count tokens in text assuming token_size characters per token."""
    if not text:  # Check if text is empty or None
        return 0
    return math.ceil(len(text) / token_size)


def process_jsonl(file_path):
    """Process JSONL file and count tokens in paragraph_text field."""
    total_tokens = 0
    line_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line_count += 1
                try:
                    data = json.loads(line.strip())
                    if 'paragraph_text' in data:
                        text = data['paragraph_text']
                        tokens = count_tokens(text)
                        total_tokens += tokens
                    else:
                        print(f"Warning: Line {line_count} missing 'paragraph_text' field")
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON at line {line_count}")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

    return total_tokens, line_count


def main():
    parser = argparse.ArgumentParser(description='Count tokens in paragraph_text fields of a JSONL file.')
    parser.add_argument('--path', '-p', required=True, help='Path to the JSONL file')

    args = parser.parse_args()

    total_tokens, line_count = process_jsonl(args.path)

    print(f"Processed {line_count} lines")
    print(f"Total tokens: {total_tokens} (assuming 4 characters per token)")


if __name__ == "__main__":
    main()