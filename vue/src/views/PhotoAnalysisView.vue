<template>
  <div class="photo_analysis">
    <h1>사진 분석 서비스</h1>

    <!-- 파일 선택 -->
    <input
      type="file"
      accept="image/png, image/jpeg"
      @change="handleFileUpload"
    />

    <!-- 에러 메시지 -->
    <p v-if="error" class="error">{{ error }}</p>

    <!-- 미리보기 -->
    <div v-if="preview">
      <h3>미리보기</h3>
      <img :src="preview" alt="preview" />
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
    },
  },
};
</script>

<style>
.photo-analysis {
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
</style>