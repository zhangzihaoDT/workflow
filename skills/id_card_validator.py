"""
身份证号校验工具

提供 `validate_id_card(id_number: str) -> bool` 用于校验中国大陆18位身份证号的合法性（含校验码 X）。

使用方式：
1) 作为库函数导入使用：
   from skills.id_card_validator import validate_id_card
   validate_id_card("310115198807175610")

2) 命令行：
   python skills/id_card_validator.py 310115198807175610 41042119810616502X
"""

from typing import Iterable


def validate_id_card(id_number: str) -> bool:
    """
    验证中国大陆 18 位身份证号（含校验码 X）。

    规则：
    - 前 17 位为数字；
    - 末位为校验码（0-9 或 X）；
    - 使用 GB 11643-1999 加权校验。

    返回 True 表示合法，否则 False。
    """
    if not isinstance(id_number, str):
        return False

    id_number = id_number.strip().upper()

    if len(id_number) != 18:
        return False

    if not id_number[:17].isdigit():
        return False

    check_digit = id_number[-1]
    if check_digit not in "0123456789X":
        return False

    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_table = ['1', '0', 'X', '9', '8', '7', '6', '5', '4', '3', '2']

    total = sum(int(num) * w for num, w in zip(id_number[:17], weights))
    expected_check = check_table[total % 11]

    return check_digit == expected_check


def batch_validate_id_cards(id_numbers: Iterable[str]) -> list[bool]:
    """批量校验，返回对应布尔结果列表。"""
    return [validate_id_card(n) for n in id_numbers]


def _demo() -> None:
    ids = [
        "310115198807175610",
        "610122199401086932",
        "520203199802171454",
        "41042119810616502X",
    ]
    for id_num in ids:
        print(f"{id_num}\t{validate_id_card(id_num)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="身份证号合法性校验（18位，含校验码）。"
    )
    parser.add_argument(
        "ids",
        nargs="*",
        help="要校验的身份证号（可传多个）",
    )
    args = parser.parse_args()

    if args.ids:
        for id_num in args.ids:
            print(f"{id_num}\t{validate_id_card(id_num)}")
    else:
        _demo()