"""
제조 결함 검사 대시보드
Track B-2: ViT + YOLOv8 통합 Streamlit 앱

실행 방법:
    streamlit run src/app.py
"""

import streamlit as st
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import io
import time
from pathlib import Path

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="제조 결함 검사 시스템",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 상수 정의 ──────────────────────────────────────────────────
DEFECT_CLASSES = ["정상 (Normal)", "스크래치 (Scratch)", "패임 (Dent)", "오염 (Contamination)"]
CLASS_COLORS = {"정상 (Normal)": "#2ecc71", "스크래치 (Scratch)": "#e74c3c",
                "패임 (Dent)": "#e67e22", "오염 (Contamination)": "#9b59b6"}


# ── 모델 로딩 (캐시) ───────────────────────────────────────────
@st.cache_resource
def load_vit_model(model_path: str):
    """ViT 분류 모델 로드 (Hugging Face Transformers)"""
    from transformers import ViTForImageClassification, ViTImageProcessor
    try:
        if Path(model_path).exists():
            processor = ViTImageProcessor.from_pretrained(model_path)
            model = ViTForImageClassification.from_pretrained(model_path)
        else:
            # 폴백: 사전 훈련 모델 (데모용)
            st.warning(f"모델 경로 없음: {model_path} → 데모 모드로 실행")
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"ViT 모델 로드 실패: {e}")
        return None, None


@st.cache_resource
def load_yolo_model(model_path: str):
    """YOLOv8 탐지 모델 로드"""
    from ultralytics import YOLO
    try:
        if Path(model_path).exists():
            model = YOLO(model_path)
        else:
            # model_path가 이미 "yolov8n.pt" 등 기본 모델명인 경우도 직접 로드
            st.warning(f"YOLOv8 경로 없음: {model_path} → 기본 모델로 실행")
            model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"YOLOv8 모델 로드 실패: {e}")
        return None


# ── 추론 함수 ──────────────────────────────────────────────────
def run_vit_inference(image: Image.Image, processor, model) -> dict:
    """ViT 분류 추론"""
    if processor is None or model is None:
        return {"class": "오류", "confidence": 0.0, "all_scores": {}}

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)[0]

    # nb02에서 훈련한 커스텀 모델의 경우 레이블 매핑
    if len(DEFECT_CLASSES) <= logits.shape[-1]:
        top_idx = probs[:len(DEFECT_CLASSES)].argmax().item()
        confidence = probs[top_idx].item()
        pred_class = DEFECT_CLASSES[top_idx]
        all_scores = {cls: probs[i].item() for i, cls in enumerate(DEFECT_CLASSES)
                      if i < len(probs)}
    else:
        top_idx = probs.argmax().item()
        confidence = probs[top_idx].item()
        pred_class = DEFECT_CLASSES[top_idx % len(DEFECT_CLASSES)]
        all_scores = {cls: float(probs[i % len(probs)]) for i, cls in enumerate(DEFECT_CLASSES)}

    return {"class": pred_class, "confidence": confidence, "all_scores": all_scores}


def run_yolo_inference(image: Image.Image, model, conf_threshold: float = 0.25) -> dict:
    """YOLOv8 탐지 추론 + 바운딩 박스 이미지 반환"""
    if model is None:
        return {"detections": [], "annotated_image": image, "count": 0}

    results = model(image, conf=conf_threshold, verbose=False)
    result = results[0]

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls_id = int(box.cls[0].item())
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
            })

    # 어노테이션 이미지
    annotated_array = result.plot()
    annotated_image = Image.fromarray(annotated_array[..., ::-1])  # BGR→RGB

    return {"detections": detections, "annotated_image": annotated_image, "count": len(detections)}


# ── UI 컴포넌트 ────────────────────────────────────────────────
def render_vit_results(vit_result: dict):
    """ViT 분류 결과 렌더링"""
    pred_class = vit_result["class"]
    confidence = vit_result["confidence"]
    color = CLASS_COLORS.get(pred_class, "#95a5a6")

    st.markdown(f"""
    <div style="background:{color}22; border-left:4px solid {color};
                padding:12px; border-radius:4px; margin:8px 0;">
        <h3 style="color:{color}; margin:0;">🏷️ {pred_class}</h3>
        <p style="margin:4px 0;">신뢰도: <strong>{confidence:.1%}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**클래스별 확률:**")
    for cls, score in vit_result.get("all_scores", {}).items():
        st.progress(score, text=f"{cls}: {score:.1%}")


def render_yolo_results(yolo_result: dict):
    """YOLOv8 탐지 결과 렌더링"""
    count = yolo_result["count"]
    status_color = "#e74c3c" if count > 0 else "#2ecc71"
    status_text = f"⚠️ {count}개 결함 탐지" if count > 0 else "✅ 결함 없음"

    st.markdown(f"""
    <div style="background:{status_color}22; border-left:4px solid {status_color};
                padding:12px; border-radius:4px; margin:8px 0;">
        <h3 style="color:{status_color}; margin:0;">{status_text}</h3>
    </div>
    """, unsafe_allow_html=True)

    st.image(yolo_result["annotated_image"], caption="YOLOv8 탐지 결과", use_container_width=True)

    if yolo_result["detections"]:
        st.markdown("**탐지된 결함 목록:**")
        for i, det in enumerate(yolo_result["detections"]):
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            st.write(f"  {i+1}. 위치: ({x1},{y1})→({x2},{y2}) | 신뢰도: {det['confidence']:.1%}")


# ── 메인 앱 ───────────────────────────────────────────────────
def main():
    # 헤더
    st.title("🔍 제조 결함 검사 대시보드")
    st.markdown("**Track B-2: ViT 분류 + YOLOv8 탐지 통합 시스템**")
    st.divider()

    # ── 사이드바 설정 ────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ 모델 설정")

        vit_model_path = st.text_input(
            "ViT 모델 경로",
            value="outputs/vit_defect_classifier",
            help="nb02에서 저장한 ViT 모델 디렉토리",
        )

        # YOLOv8 모델 선택 (사전 정의 모델 selectbox)
        model_choice = st.selectbox(
            "YOLOv8 모델 선택",
            ["YOLOv8n (초경량, 추론 ↑)", "YOLOv8s (소형, 균형)"],
            index=0,
            help="YOLOv8n: 가장 빠른 추론 속도 / YOLOv8s: 속도와 정확도 균형",
        )
        model_map = {
            "YOLOv8n (초경량, 추론 ↑)": "yolov8n.pt",
            "YOLOv8s (소형, 균형)": "yolov8s.pt",
        }
        selected_model = model_map[model_choice]

        yolo_model_path = st.text_input(
            "YOLOv8 모델 경로 (파인튜닝 시 직접 입력)",
            value=f"outputs/yolo_defect/weights/best.pt",
            help="nb03에서 훈련한 YOLOv8 가중치. 없으면 위에서 선택한 기본 모델 사용",
        )
        # 파인튜닝 가중치가 없을 경우 selectbox 선택 모델로 폴백
        _effective_yolo = yolo_model_path if Path(yolo_model_path).exists() else selected_model
        st.caption(f"실제 사용 모델: `{_effective_yolo}`")

        conf_threshold = st.slider(
            "YOLOv8 신뢰도 임계값", min_value=0.1, max_value=0.9, value=0.25, step=0.05
        )

        st.divider()
        st.markdown("### 시스템 정보")
        device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
        st.info(f"디바이스: {device}")

        load_models_btn = st.button("🔄 모델 로드", type="primary", use_container_width=True)

    # ── 모델 로딩 ────────────────────────────────────────────
    if load_models_btn or "models_loaded" not in st.session_state:
        with st.spinner("모델 로딩 중..."):
            vit_processor, vit_model = load_vit_model(vit_model_path)
            yolo_model = load_yolo_model(_effective_yolo)
            st.session_state.vit_processor = vit_processor
            st.session_state.vit_model = vit_model
            st.session_state.yolo_model = yolo_model
            st.session_state.models_loaded = True

    # ── 이미지 업로드 ─────────────────────────────────────────
    st.subheader("📁 검사 이미지 업로드")
    uploaded_file = st.file_uploader(
        "제품 이미지를 업로드하세요 (JPG, PNG, BMP)",
        type=["jpg", "jpeg", "png", "bmp"],
        help="제조 라인에서 촬영한 제품 표면 이미지",
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(image, caption=f"업로드: {uploaded_file.name}", use_container_width=True)
        with col_info:
            st.metric("이미지 크기", f"{image.width} × {image.height}")
            st.metric("파일 크기", f"{uploaded_file.size / 1024:.1f} KB")

        # ── 분석 실행 ────────────────────────────────────────
        if st.button("🚀 결함 검사 실행", type="primary", use_container_width=True):
            start_time = time.time()

            col_vit, col_yolo = st.columns(2)

            with col_vit:
                st.subheader("🧠 ViT 분류 결과")
                with st.spinner("ViT 추론 중..."):
                    vit_result = run_vit_inference(
                        image,
                        st.session_state.get("vit_processor"),
                        st.session_state.get("vit_model"),
                    )
                render_vit_results(vit_result)

            with col_yolo:
                st.subheader("📦 YOLOv8 탐지 결과")
                with st.spinner("YOLOv8 추론 중..."):
                    yolo_result = run_yolo_inference(
                        image,
                        st.session_state.get("yolo_model"),
                        conf_threshold,
                    )
                render_yolo_results(yolo_result)

            elapsed = time.time() - start_time
            st.success(f"✅ 분석 완료 ({elapsed:.2f}초)")

            # 검사 판정
            is_defective = (vit_result["class"] != "정상 (Normal)") or (yolo_result["count"] > 0)
            if is_defective:
                st.error("🚨 **판정: 불량 (FAIL)** — 제품을 제거하고 재검사하세요.")
            else:
                st.success("✅ **판정: 양호 (PASS)** — 제품이 품질 기준을 충족합니다.")
    else:
        st.info("👆 이미지를 업로드하면 자동으로 결함 검사를 시작합니다.")

        # 샘플 데모 표시
        st.markdown("---")
        st.markdown("### 📌 사용 안내")
        st.markdown("""
        1. **왼쪽 사이드바**에서 모델 경로를 설정하세요
            - ViT 모델: `nb02_vit_transfer_learning.ipynb` 에서 저장한 경로
            - YOLOv8 모델: `nb03_yolov8_finetuning.ipynb` 에서 저장한 경로
        2. **이미지 업로드** 영역에 제품 사진을 드래그하거나 클릭하여 업로드
        3. **결함 검사 실행** 버튼을 클릭하면 ViT + YOLOv8이 동시에 분석
        4. **판정 결과** (PASS/FAIL)를 확인하세요
        """)


if __name__ == "__main__":
    main()