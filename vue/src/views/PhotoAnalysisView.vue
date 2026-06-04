<template>
  <div class="photo-page">
    <h2 class="page-title">사진 분석 서비스</h2>
    <p class="page-subtitle">거리 사진을 업로드하여 안전도를 분석하고 위험 요소를 확인하세요.</p>

    <div class="upload-section" >
      <div class="upload-card">
        <h3 class="card-heading">이미지 업로드</h3>
        <p class="card-desc">JPG 또는 PNG 형식의 거리 이미지를 업로드해주세요.</p>
          <label class="upload-box" @dragover.prevent @drop.prevent="handleDrop">
            <input
              type="file"
              accept="image/png, image/jpeg"
              @change="handleFileUpload"
              hidden
            />
            <div class="upload-icon">⬆️</div>
            <div class="upload-text">이미지를 드래그하거나 클릭하여 선택</div>
            <div class="upload-hint">JPG, PNG (최대 10MB)</div>
          </label>
        <p v-if="error" class="error">{{ error }}</p>
      </div>

      <div class="preview-card">
        <div v-if="preview" class="preview-wrapper">
          <img :src="preview" alt="preview" class="preview-img" />
        </div>
        <div v-else class="preview-placeholder">
          <div class="placeholder-icon">🖼️</div>
          <div>이미지를 업로드하여 분석을 시작하세요</div>
        </div>
      </div>
    </div>

    <p v-if="loading" class="loading-text">분석 중...</p>

    <div v-if="safetyScore !== null" class="result-container">
      <h3>분석 결과</h3>
      <p class="score-text">안전 점수: <span>{{ safetyScore.toFixed(2) }}</span> / 5.00</p>

      <div v-if="explanation" class="xai-box">
        <h4>💡 AI 분석 근거 설명</h4>
        <p v-html="explanation"></p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      file: null,
      preview: null,
      error: "",
      safetyScore: null,
      explanation: "",
      loading: false,
    };
  },

  methods: {
    isValidImage(file) {
      const allowedTypes = ["image/png", "image/jpeg"];
      return allowedTypes.includes(file.type);
    },

    setImageFile(file) {
      if (!file) return;

      if (!this.isValidImage(file)) {
        this.error = "PNG 또는 JPG 파일만 업로드 가능합니다.";
        this.file = null;
        this.preview = null;
        return;
      }

      this.error = "";
      this.file = file;
      this.preview = URL.createObjectURL(file);

      this.uploadImage();
    },

    handleFileUpload(event) {
      const file = event.target.files[0];
      this.setImageFile(file);
    },

    handleDrop(event) {
      const file = event.dataTransfer.files[0];
      this.setImageFile(file);
    },

    async uploadImage() {
      if (!this.file) return;

      this.loading = true;
      this.safetyScore = null;
      this.explanation = "";
      this.error = "";

      const formData = new FormData();
      formData.append("file", this.file);

      try {
        const response = await fetch("http://127.0.0.1:8000/predict-upload", {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          throw new Error("분석 요청 실패");
        }

        const result = await response.json();
        this.safetyScore = Number(result.safety_score);
        this.explanation = result.explanation || "";
      } catch (err) {
        this.error = "사진 분석 중 오류가 발생했습니다.";
        console.error(err);
      } finally {
        this.loading = false;
      }
    },
  },
};
</script>

<style>
.photo-page {
  display: flex;
  flex-direction: column;
}

.page-title {
  font-size: 24px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 6px;
}

.page-subtitle {
  font-size: 15px;
  color: #64748b;
  margin-bottom: 24px;
}

.upload-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
  margin-bottom: 24px;
}

.upload-card,
.preview-card,

.card-heading {
  font-size: 17px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 4px;
}

.card-desc {
  font-size: 14px;
  color: #64748b;
  margin-bottom: 16px;
}

.upload-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  border: 2px dashed #cbd5e1;
  border-radius: 10px;
  padding: 40px 20px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.upload-box:hover {
  border-color: #3b82f6;
  background: #f8fafc;
}

.upload-icon {
  font-size: 28px;
}

.upload-text {
  font-size: 15px;
  font-weight: 600;
  color: #334155;
}

.upload-hint {
  font-size: 13px;
  color: #94a3b8;
}

.preview-card {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
}

.preview-wrapper {
  width: 100%;
  display: flex;
  justify-content: center;
}

.preview-img {
  max-width: 100%;
  max-height: 280px;
  border-radius: 10px;
  border: 1px solid #e2e8f0;
}

.preview-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  color: #94a3b8;
  font-size: 14px;
}

.placeholder-icon {
  font-size: 36px;
  opacity: 0.5;
}

.error {
  color: #ef4444;
  margin-top: 12px;
  font-size: 14px;
}

.loading-text {
  color: #3b82f6;
  font-weight: 600;
  margin-bottom: 16px;
}

.result-container {
  margin-bottom: 24px;
  padding: 20px;
  background: white;
  border-radius: 12px;
  border: 1px solid #e2e8f0;
}

.score-text span {
  font-size: 20px;
  font-weight: bold;
  color: #2c3e50;
}

.xai-box {
  margin-top: 15px;
  padding: 15px;
  background-color: #f4f6f9;
  border-left: 4px solid #3498db;
  border-radius: 4px;
  text-align: left;
}

.xai-box h4 {
  margin: 0 0 8px 0;
  color: #2c3e50;
}

.xai-box p {
  margin: 0;
  font-size: 14px;
  color: #555;
  line-height: 1.6;
}
</style>