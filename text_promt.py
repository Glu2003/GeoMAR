import os
import sys
import glob
import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 命令行参数处理 =================
if len(sys.argv) != 4:
    print(f"Usage: {sys.argv[0]} <rgb_dir> <parsing_map_dir> <output_dir>")
    print(f"Example: python {sys.argv[0]} ./images_rgb ./images_parsing ./output_captions")
    sys.exit(1)

rgb_dir = sys.argv[1]
map_dir = sys.argv[2]
output_dir = sys.argv[3]
os.makedirs(output_dir, exist_ok=True)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
rgb_files = sorted([p for p in glob.glob(os.path.join(rgb_dir, "*"))
                    if p.lower().endswith(IMG_EXTS)])

if not rgb_files:
    print(f"No images found in {rgb_dir}")
    sys.exit(0)

# ================= 模型加载 =================
MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
print(f"Loading model: {MODEL_ID}...")

try:
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,  # FP16加速
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
except Exception:
    print("Flash Attention 2 not available, fallback to default...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

processor = AutoProcessor.from_pretrained(MODEL_ID)
model.eval()
torch.set_grad_enabled(False)  # 禁用梯度
# ================= System Prompt 定义 =================
SYSTEM_PROMPT = """You are an expert visual analyst for high-fidelity face restoration. 
You will receive two images:
1. **Image 1 (RGB)**: Source for color, texture, lighting, and fine details.
2. **Image 2 (Parsing Map)**: Source for GEOMETRY (shapes, open/closed states). The map is the ground truth for structure.

Your task is to generate **6 separate, dense descriptive phrases**. 
**CRITICAL**: Do not use full sentences with filler words (e.g., "The nose is...", "She has..."). Start directly with adjectives or nouns. Focus heavily on **texture and lighting** from the RGB image.

### ATTRIBUTE RULES & SOURCES:

**1. [Global]: (Source: RGB)**
   - **Content**: Combine age, gender, race, emotion, pose, skin_tone, wrinkles, hair_color, hair_style, accessories.
   - **Style**: Use comma-separated descriptors.
   - **ADDITION**: Also describe the **Lighting Quality** (e.g., soft lighting, cinematic, harsh shadows) to set the global atmosphere.
   - **Constraint**: Keep it high-level. Do not describe eyes/nose/mouth details here.
   - **Enums**:
     - age: teen/young/middle_aged/senior
     - gender: male/female
     - race: asian/indian/black/white/middle_eastern/latino_hispanic
     - skin_tone: pale/normal/dark
     - wrinkles: none/mild/heavy
     - hair_color: black/blond/brown/gray
     - hair_style: long/short/bald/bangs 
     - accessories: necklace/hat/necktie/earrings/eyeglasses/none

**2. [Eyes]:** 
   - **Geometry (Map)**: Integrate 'shape' (narrow/normal/round) & 'state' (closed/slightly_open/open).
   - **Texture (RGB)**: Describe iris color, eyelashes, and specular reflections.
   - **Constraint**: If shape is 'normal', focus purely on the color and light (e.g., "Open, bright hazel eyes with sharp highlights").

**3. [Nose]:**
   - **Geometry (Map)**: Integrate 'shape' (small/normal/big/pointy).
   - **Texture (RGB)**: Describe skin pores and highlights on the tip.

**4. [Mouth]:**
   - **Geometry (Map)**: Integrate 'size' (thin/normal/thick) & 'state' (closed/slightly_open/open).
   - **Detail (RGB)**: Describe lip texture (dry/glossy/cracked), lipstick color, and 'teeth_visible' (yes/no).

**5. [Eyebrows]:**
   - **Geometry (Map)**: Integrate 'shape' (arched/bushy/normal).
   - **Texture (RGB)**: Describe hair density (sparse/dense) and grooming.

**6. [Face]:**
   - **Geometry (Map)**: Integrate 'shape' (oval/round/square/heart/normal).
   - **Detail (RGB)**: Describe 'moles', 'freckles', 'facial_hair', and 'makeup'.
   - **Enums**: moles (none/few/many), freckles (none/light/heavy), facial_hair (mustache/goatee/sideburns/stubble/none), makeup (none/light/heavy).

### OUTPUT FORMAT:
Output exactly 6 lines. Each line starts with the tag `[Tag]:` followed by a concise, dense phrase.

**Example of Desired Quality:**
[Global]: Young asian female, pale skin, heavy makeup, neutral expression, facing front, long black hair with bangs.
[Eyes]: Narrow, open eyes featuring dark brown irises and thick eyelashes.
[Nose]: Small nose with smooth skin texture and a soft highlight on the tip.
[Mouth]: Slightly open mouth with thick, glossy red lips, teeth visible.
[Eyebrows]: Arched, neatly groomed black eyebrows.
[Face]: Oval face shape with no moles or freckles, heavy makeup finish.
"""

# ================= 批量参数 =================
BATCH_SIZE =16  # 根据显存调整

print(f"Starting processing of {len(rgb_files)} images in batches of {BATCH_SIZE}...")

for i in tqdm(range(0, len(rgb_files), BATCH_SIZE), desc="Processing batches", unit="batch"):
    batch_files = rgb_files[i:i+BATCH_SIZE]
    batch_messages = []
    batch_basenames = []
    
    for rgb_path in batch_files:
        filename = os.path.basename(rgb_path)
        basename = os.path.splitext(filename)[0]
        batch_basenames.append(basename)

        # 找对应 map
        map_path = None
        for ext in IMG_EXTS:
            candidate = os.path.join(map_dir, basename + ext)
            if os.path.exists(candidate):
                map_path = candidate
                break
        if not map_path:
            tqdm.write(f"Skipping {filename}: Map not found.")
            continue

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": rgb_path},
                {"type": "image", "image": map_path},
                {"type": "text", "text": "Analyze Image 1 (RGB) and Image 2 (Parsing Map) to generate the 6 descriptive sentences."}
            ]}
        ]
        batch_messages.append(messages)

    if not batch_messages:
        continue

    try:
        # 预处理
        texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]
        images_list = [process_vision_info(msg)[0] for msg in batch_messages]  # 只用 image_inputs
        inputs = processor(
            text=texts,
            images=images_list,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # 生成
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False
        )

        # 解码
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        outputs = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 保存
        for basename, output_text in zip(batch_basenames, outputs):
            txt_out_path = os.path.join(output_dir, basename + ".txt")
            with open(txt_out_path, "w", encoding="utf-8") as f:
                f.write(output_text)

        tqdm.write(f"Processed batch {i//BATCH_SIZE + 1} / {len(rgb_files)//BATCH_SIZE + 1}")

    except Exception as e:
        tqdm.write(f"Error processing batch starting with {batch_files[0]}: {e}")
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()

print(f"\nDone! Results saved to {output_dir}")