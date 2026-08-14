"""
AES 加密模块 — 保护 API Key 等敏感配置
使用 Fernet (AES-128-CBC + HMAC-SHA256)
"""

import os
import base64
import json
from pathlib import Path
from cryptography.fernet import Fernet

# 密钥存放位置（优先环境变量，其次文件）
SECRET_ENV = "AGENT_SECRET"
SECRET_FILE = Path.home() / ".trading_agent" / "secret.key"


def _get_or_create_key() -> bytes:
    """获取或生成 Fernet 密钥"""
    env = os.environ.get(SECRET_ENV)
    if env:
        return env.encode()

    if SECRET_FILE.exists():
        return SECRET_FILE.read_bytes().strip()

    # 首次运行：生成密钥
    key = Fernet.generate_key()
    SECRET_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRET_FILE.write_bytes(key)
    # 限制权限（Unix）
    try:
        os.chmod(SECRET_FILE, 0o600)
    except OSError:
        pass  # Windows 不支持
    print(f"[crypto] 新密钥已生成: {SECRET_FILE}")
    print(f"[crypto] 也可设置环境变量: export {SECRET_ENV}={key.decode()}")
    return key


def get_fernet() -> Fernet:
    return Fernet(_get_or_create_key())


def encrypt_value(plaintext: str) -> str:
    """加密字符串，返回 base64 密文"""
    return get_fernet().encrypt(plaintext.encode()).decode()


def decrypt_value(ciphertext: str) -> str:
    """解密 base64 密文"""
    return get_fernet().decrypt(ciphertext.encode()).decode()


def encrypt_config_file(config_path: str, fields: list[str]):
    """
    将 YAML 配置文件中的指定字段加密（原地替换）
    已加密的字段会跳过（以 'ENC:' 开头标识）

    Args:
        config_path: YAML 文件路径
        fields: 需要加密的字段名列表
    """
    import yaml

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(config_path)

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fernet = get_fernet()
    changed = False

    def _encrypt_dict(d, field_list):
        nonlocal changed
        for key, val in d.items():
            if isinstance(val, dict):
                _encrypt_dict(val, field_list)
            elif key in field_list and isinstance(val, str) and not val.startswith("ENC:"):
                d[key] = "ENC:" + fernet.encrypt(val.encode()).decode()
                changed = True
                print(f"  [加密] {key}")

    _encrypt_dict(cfg, fields)

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)
        print(f"[crypto] 配置已加密保存: {config_path}")
    else:
        print("[crypto] 所有字段已加密，无需修改")


def decrypt_config_value(val: str) -> str:
    """如果值以 ENC: 开头则解密，否则原样返回"""
    if isinstance(val, str) and val.startswith("ENC:"):
        return decrypt_value(val[4:])
    return val


if __name__ == "__main__":
    # 测试
    secret = "my_broker_api_key_12345"
    enc = encrypt_value(secret)
    dec = decrypt_value(enc)
    print(f"原文: {secret}")
    print(f"密文: {enc}")
    print(f"解密: {dec}")
    assert dec == secret
    print("✓ 加密解密正常")
