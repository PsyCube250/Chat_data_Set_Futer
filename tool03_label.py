"""Tool 03 — Human calibration.

Input: Tool 02 candidate samples.
Output: human-verified labels used to calibrate later selection.

Purification level: 1 / Foundation.
"""

import argparse
import os
import random

import config
import utils


def make_key(rec: dict) -> str:
    return f'{rec["conversation_id"]}|{rec["exchange_index"]}'


def load_classified(classified_path: str) -> dict:
    """加载 classified.json，构建 (conv_id|idx) -> labels 列表 的查找表。"""
    if not os.path.exists(classified_path):
        return {}
    data = utils.load_json(classified_path)
    lookup = {}
    for item in data:
        key = f'{item["conversation_id"]}|{item["exchange_index"]}'
        lookup[key] = item.get("labels", [])
    return lookup


def save_taxonomy(taxonomy: dict):
    """将更新后的 taxonomy 写回文件。"""
    utils.save_json(taxonomy, config.TAXONOMY_PATH)


def run(candidates_path: str, output_path: str, classified_path: str = None, manual: bool = False):
    candidates = utils.load_json(candidates_path)
    taxonomy = utils.load_taxonomy(config.TAXONOMY_PATH)
    dim_names = list(taxonomy.keys())

    # 如果不是手动模式，加载预测标签
    pred_lookup = {}
    if not manual and classified_path:
        pred_lookup = load_classified(classified_path)

    # 加载已有标注
    labels = {}
    if os.path.exists(output_path):
        labels = {item["key"]: item for item in utils.load_json(output_path)}
        print(f"检测到已有 {len(labels)} 条标注记录，将自动跳过这些")

    todo = [c for c in candidates if make_key(c) not in labels]
    random.shuffle(todo)

    print(f"共有 {len(candidates)} 条候选样本，待标注 {len(todo)} 条")
    print("\n当前维度表:")
    for i, dim in enumerate(dim_names, 1):
        print(f"  {i}) {dim} — {taxonomy[dim]}")
    print("  0) 都不体现 / 不属于以上任何维度")
    print(f"  (共 {len(dim_names)} 个维度)\n")

    if manual:
        print("【纯手动模式】请自行输入维度编号（逗号分隔多选），0=无，-跳过，q退出")
    else:
        print("【自动建议模式】直接回车采纳建议，或输入数字修改")

    print("操作说明：")
    print("  - 输入数字（如 5 或 5,7）：指定维度（多选用逗号分隔）")
    print("  - 输入 0：表示无标签")
    print("  - 输入 - ：跳过本条（不标注）")
    print("  - 输入 +新维度名（如 +幽默感）：新增一个维度并标注当前条")
    print("  - 输入 list 或 l：重新显示当前完整维度表")
    print("  - 输入 * ：将当前标签应用于后续所有待标注条（逐一确认）")
    print("  - 输入 q ：保存并退出\n")

    done_count = len([v for v in labels.values() if v.get("human_dims")])

    try:
        idx = 0
        while idx < len(todo):
            rec = todo[idx]
            key = make_key(rec)

            # 获取模型预测标签（仅非手动模式）
            pred_dims = []
            pred_str = ""
            if not manual and classified_path:
                pred_dims = [lbl["dim"] for lbl in pred_lookup.get(key, [])]
                if not pred_dims:
                    if rec.get("model_dim") and rec["model_dim"] in dim_names:
                        pred_dims = [rec["model_dim"]]
                # 转换为序号
                pred_indices = []
                for d in pred_dims:
                    if d in dim_names:
                        pred_indices.append(str(dim_names.index(d) + 1))
                pred_str = ",".join(pred_indices) if pred_indices else "（无建议）"

            print("-" * 60)
            if rec.get("context_summary"):
                print(f'背景: {rec["context_summary"]}')
            if rec.get("prior_other_text"):
                print(f'对方: {rec["prior_other_text"]}')
            print(f'目标本人: {rec["target_text"]}')
            if not manual and classified_path:
                print(f'[模型建议标签: {pred_str}]')

            raw = input("你的判断 > ").strip()

            if raw == "q":
                break
            if raw == "-":
                idx += 1
                continue
            if raw in ("list", "l"):
                print("\n当前维度表:")
                for i, dim in enumerate(dim_names, 1):
                    print(f"  {i}) {dim} — {taxonomy[dim]}")
                print(f"  (共 {len(dim_names)} 个维度)\n")
                continue

            # ---- 新增维度 ----
            if raw.startswith("+"):
                new_dim = raw[1:].strip()
                if not new_dim:
                    print("  请提供维度名称，如 +幽默感")
                    continue
                if new_dim in dim_names:
                    print(f"  '{new_dim}' 已存在，使用现有维度")
                    human_dims = [new_dim]
                else:
                    # 添加到 taxonomy
                    # 默认描述：让用户输入
                    desc = input(f"  请输入 '{new_dim}' 的简短描述（直接回车使用默认描述'待完善'）> ").strip()
                    if not desc:
                        desc = "待完善"
                    taxonomy[new_dim] = desc
                    dim_names.append(new_dim)
                    save_taxonomy(taxonomy)
                    print(f"  ✓ 已添加新维度：{new_dim}（编号 {len(dim_names)}）")
                    human_dims = [new_dim]
                # 直接标注当前条
                labels[key] = {
                    "key": key,
                    "conversation_id": rec["conversation_id"],
                    "exchange_index": rec["exchange_index"],
                    "target_text": rec["target_text"],
                    "model_dim": rec.get("model_dim", ""),
                    "model_confidence": rec.get("model_confidence", 0),
                    "human_dims": human_dims,
                }
                utils.save_json(list(labels.values()), output_path)
                done_count += 1
                print("  ✓ 已标注")
                if done_count == config.LABEL_TARGET_MIN:
                    print(f"\n已标注 {done_count} 条，达到最低目标，之后可以随时用 q 停下\n")
                if done_count >= config.LABEL_TARGET_MAX:
                    print(f"\n已达到 {config.LABEL_TARGET_MAX} 条上限，建议停止标注了\n")
                    break
                idx += 1
                continue

            # ---- 批量操作 ----
            if raw.startswith("*"):
                if len(raw) > 1:
                    parts = raw[1:].strip()
                    if parts:
                        try:
                            idxs = [int(x.strip()) for x in parts.split(",") if x.strip()]
                            batch_dims = [dim_names[i - 1] for i in idxs if 1 <= i <= len(dim_names)]
                            if not batch_dims:
                                print("  标签无效，取消批量操作")
                                continue
                        except (ValueError, IndexError):
                            print("  标签格式错误，取消批量操作")
                            continue
                    else:
                        if not manual and classified_path and pred_dims:
                            batch_dims = pred_dims
                        else:
                            print("  当前没有建议标签，请手动输入标签（如 *5,7）")
                            continue
                else:
                    if not manual and classified_path and pred_dims:
                        batch_dims = pred_dims
                    else:
                        print("  当前没有建议标签，请手动输入标签（如 *5,7）")
                        continue

                if not batch_dims:
                    print("  没有有效的标签，取消批量操作")
                    continue

                print(f"将应用标签 {batch_dims} 给后续所有待标注条（共 {len(todo)-idx-1} 条）")
                confirm = input("确认批量应用吗？(y/N) > ").strip().lower()
                if confirm != "y":
                    print("取消批量操作")
                    continue

                batch_count = 0
                for j in range(idx + 1, len(todo)):
                    next_key = make_key(todo[j])
                    if next_key in labels:
                        continue
                    print("-" * 30)
                    print(f"[{j+1}/{len(todo)}] {todo[j].get('target_text', '')[:80]}...")
                    keep = input("保留此标签？(y/n/其他跳过) > ").strip().lower()
                    if keep == "y":
                        labels[next_key] = {
                            "key": next_key,
                            "conversation_id": todo[j]["conversation_id"],
                            "exchange_index": todo[j]["exchange_index"],
                            "target_text": todo[j]["target_text"],
                            "model_dim": todo[j].get("model_dim", ""),
                            "model_confidence": todo[j].get("model_confidence", 0),
                            "human_dims": batch_dims,
                        }
                        done_count += 1
                        batch_count += 1
                        print("  ✓ 已标注")
                    else:
                        print("  跳过本条")
                print(f"批量操作完成，共标注 {batch_count} 条")
                # 当前条也标注
                if key not in labels:
                    labels[key] = {
                        "key": key,
                        "conversation_id": rec["conversation_id"],
                        "exchange_index": rec["exchange_index"],
                        "target_text": rec["target_text"],
                        "model_dim": rec.get("model_dim", ""),
                        "model_confidence": rec.get("model_confidence", 0),
                        "human_dims": batch_dims,
                    }
                    done_count += 1
                    print("  ✓ 当前条也已标注")
                utils.save_json(list(labels.values()), output_path)
                idx += 1
                continue

            # ---- 正常处理当前条 ----
            if raw == "":
                if not manual and classified_path and pred_dims:
                    human_dims = pred_dims
                else:
                    print("  请输入数字或 0，不能为空")
                    continue
            elif raw == "0":
                human_dims = []
            else:
                try:
                    idxs = [int(x.strip()) for x in raw.split(",") if x.strip()]
                    human_dims = [dim_names[i - 1] for i in idxs if 1 <= i <= len(dim_names)]
                except (ValueError, IndexError):
                    print("  输入没看懂，请重新输入")
                    continue

            labels[key] = {
                "key": key,
                "conversation_id": rec["conversation_id"],
                "exchange_index": rec["exchange_index"],
                "target_text": rec["target_text"],
                "model_dim": rec.get("model_dim", ""),
                "model_confidence": rec.get("model_confidence", 0),
                "human_dims": human_dims,
            }
            utils.save_json(list(labels.values()), output_path)
            done_count += 1

            if done_count == config.LABEL_TARGET_MIN:
                print(f"\n已标注 {done_count} 条，达到最低目标，之后可以随时用 q 停下\n")
            if done_count >= config.LABEL_TARGET_MAX:
                print(f"\n已达到 {config.LABEL_TARGET_MAX} 条上限，建议停止标注了\n")
                break

            idx += 1

    except KeyboardInterrupt:
        print("\n\n已中断，进度已保存，下次运行会自动跳过标过的部分")

    print(f"\n本次共标注 {done_count} 条，结果保存在 {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="Tool2 输出的 candidates.json 路径")
    parser.add_argument("--output", required=True, help="标注结果输出路径")
    parser.add_argument("--classified", help="Tool2 输出的 classified.json 路径（用于自动建议模式）")
    parser.add_argument("--manual", action="store_true", help="纯手动模式，不显示模型建议（不需要 --classified）")
    args = parser.parse_args()
    run(args.candidates, args.output, args.classified, args.manual)