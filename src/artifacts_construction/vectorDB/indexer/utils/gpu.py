import torch


def get_gpu_vram_size(): 
    if not torch.cuda.is_available():
        return 0.0
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
    return total_memory


VRAM_BATCH_TABLE = [
    (120, 512),
    (80, 128),
    (40, 64),
    (24, 48),
    (16, 16),
    (12, 8),
    (0, 4),   # fallback
]

def get_optimal_max_batch_size(vram_gb: float) -> int: # GPU VRAM 크기에 따라 최적의 최대 배치 크기를 반환
    for min_vram, batch_size in VRAM_BATCH_TABLE:
        if vram_gb >= min_vram:
            return batch_size


TOKEN_BATCH_TABLE_BY_VRAM = [
    # VRAM ≥ 120GB
    (120, [
        (20000, 8), # (Token Size, Batch Size)
        (15000, 16),
        (12000, 32),
        (10000, 64),
        (7000, 96),
        (5000, 128),
        (3000, 192),
        (2000, 256),
        (1000, 384),
        (0, 512),
    ]),

    # VRAM ≥ 40GB
    (40, [
        (12000, 4),
        (8000, 8),
        (5000, 16),
        (3000, 24),
        (2000, 32),
        (1000, 48),
        (0, 64),
    ]),

    # VRAM ≥ 24GB
    (24, [
        (10000, 16),
        (7000, 24),
        (5000, 40),
        (3000, 48),
        (2000, 64),
        (1000, 80),
        (0, 96),
    ]),

    # VRAM < 24GB
    (0, [
        (10000, 1),
        (7000, 2),
        (4000, 3),
        (2000, 4),
        (1000, 6),
        (0, None),
    ]),
]


def lookup_batch_size(token_count: int, rules: list, default: int) -> int:
    for min_tokens, batch_size in rules:
        if token_count >= min_tokens:
            return default if batch_size is None else batch_size


def get_dynamic_batch_size(
    token_count: int,
    vram_gb: float,
    max_batch_size: int = 8
) -> int:
    # 1단계: VRAM 기반 최대 batch
    vram_based_max = get_optimal_max_batch_size(vram_gb)
    effective_max = min(max_batch_size, vram_based_max)

    # 2단계: VRAM 구간 선택
    for min_vram, token_rules in TOKEN_BATCH_TABLE_BY_VRAM:
        if vram_gb >= min_vram:
            token_based = lookup_batch_size(
                token_count,
                token_rules,
                default=max_batch_size
            )
            break

    # 3단계: 최종 batch size
    final_batch_size = min(token_based, effective_max)
    return max(1, final_batch_size)


def get_gpu_memory_info(device_id: int = 0): # 현재 GPU 메모리 사용량 정보 반환 (GB 단위)
    if not torch.cuda.is_available():
        return None, None, None
    
    allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)  # GB
    reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)    # GB
    total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)  # GB
    
    return allocated, reserved, total
