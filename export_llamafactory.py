"""Export final records to a LLaMA-Factory-compatible dataset format."""

import argparse

import utils

SKIP_NO_CONTEXT = True


def run(input_path: str, output_path: str):
    data = utils.load_json(input_path)
    records = data.get("records", data if isinstance(data, list) else [])

    alpaca_data = []
    skipped = 0
    for rec in records:
        prior = rec.get("prior_other_text")
        if not prior:
            if SKIP_NO_CONTEXT:
                skipped += 1
                continue
            prior = ""
        alpaca_data.append(
            {
                "instruction": prior,
                "input": "",
                "output": rec["target_text"],
                "system": rec.get("context_summary", ""),
            }
        )

    utils.save_json(alpaca_data, output_path)
    print(f"转换完成：{len(alpaca_data)} 条（跳过 {skipped} 条无上下文的），输出到 {output_path}")
    print()
    print("接下来手动操作：")
    print(f"1. 把 {output_path} 复制到 LLaMA-Factory 项目的 data/ 目录下")
    print("2. 打开 LLaMA-Factory/data/dataset_info.json，加一条（跟已有条目平级）：")
    print(
        '   "persona_clone": {\n'
        f'     "file_name": "{output_path.split("/")[-1]}"\n'
        "   }"
    )
    print("3. WebUI 里数据集下拉框刷新后应该能看到 persona_cloneV2，选它做 LoRA SFT 训练")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Tool4 产出的 final_dataset.json")
    parser.add_argument("--output", required=True, help="转换后的 alpaca 格式文件名，比如 persona_data.json")
    args = parser.parse_args()
    run(args.input, args.output)
