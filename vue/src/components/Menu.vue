<template>
  <div class="menu">
    <button @click="toggleMenu">메뉴</button>

    <div v-if="isOpen" class="dropdown">
      <router-link to="/">지도</router-link>
      <router-link to="/photo_analysis">사진 분석 서비스</router-link>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      isOpen: false,
    };
  },
  methods: {
    toggleMenu() {
      this.isOpen = !this.isOpen;
    },
  },
};
</script>

<style>
.menu {
  position: relative;
  margin-bottom: 12px;
}

.menu button {
  padding: 8px 14px;
  cursor: pointer;
}

.dropdown {
  position: absolute;
  top: 38px;
  left: 0;
  width: 180px;
  background: white;
  border: 1px solid #ddd;
  z-index: 10;
  display: flex;
  flex-direction: column;
}

.dropdown a {
  padding: 10px;
  text-decoration: none;
  color: black;
}

.dropdown a:hover {
  background: #f2f2f2;
}
</style>