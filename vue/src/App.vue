<template>
  <div id="app-wrapper">
    <Menu />
    <main class="page-content">
      <router-view />
    </main>
  </div>
</template>

<script>
import Menu from "./components/Menu.vue";

export default {
  components: {
    Menu,
  },
};
</script>

<style>
* {
  box-sizing: border-box;
  margin: 0;
  padding: 0;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Pretendard",
    "Noto Sans KR", sans-serif;
  background-color: #f8fafc;
  color: #1e293b;
  -webkit-font-smoothing: antialiased;
}

#app-wrapper {
  min-height: 100vh;
  background-color: #f8fafc;
}

.page-content {
  padding: 32px 48px;
  max-width: 1400px;
  margin: 0 auto;
}
</style>