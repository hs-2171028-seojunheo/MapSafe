<template>
  <div class="photo_analysis">
    <h1>사진 분석 서비스</h1>

    <input
      type="file"
      accept="image/png, image/jpeg"
      @change="handleFileUpload"
    />

    <p v-if="error" class="error">{{ error }}</p>

    <div v-if="preview">
      <h3>미리보기</h3>
      <img :src="preview" alt="preview" />
    </div>
  </div>
  
  <p v-if="loading">분석 중...</p>

  <div v-if="safetyScore !== null" class="result-container">
    <h3>분석 결과</h3>
    <p class="score-text">안전 점수: <span>{{ safetyScore.toFixed(2) }}</span> / 5.00</p>
    
    <div v-if="explanation" class="xai-box">
      <h4>💡 AI 분석 근거 설명</h4>
      <p v-html="explanation"></p>
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
    handleFileUpload(event) {
      const selectedFile = event.target.files[0];

      if (!selectedFile) return;

      // 파일 타입 검사
      const validTypes = ["image/png", "image/jpeg"];

      if (!validTypes.includes(selectedFile.type)) {
        this.error = "PNG 또는 JPG 파일만 업로드 가능합니다.";
        this.file = null;
        this.preview = null;
        return;
      }

      this.error = "";
      this.file = selectedFile;

      // 미리보기 생성
      this.preview = URL.createObjectURL(selectedFile);
      this.uploadImage();
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
        this.safetyScore = result.safety_score;
        this.explanation = result.explanation;
        
      } catch (err) {
        this.error = "사진 분석 중 오류가 발생했습니다.";
        console.error(err);
      } finally {
        this.loading = false;
      }
    }
  },
};
</script>

<style>
.photo_analysis {
  padding: 20px;
}

.error {
  color: red;
  margin-top: 10px;
}

img {
  margin-top: 10px;
  max-width: 300px;
  border: 1px solid #ddd;
}

.result-container {
  margin-top: 20px;
  padding: 15px;
  background: #fdfdfd;
  border-radius: 8px;
  border: 1px solid #eaeaea;
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