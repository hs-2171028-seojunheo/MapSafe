<template>
  <header class="app-header">
    <div class="logo">
      <span class="logo-icon">
        <svg width="26" height="26" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2L4 5V11C4 16 7.5 20.5 12 22C16.5 20.5 20 16 20 11V5L12 2Z" fill="#2563eb"/>
          <path d="M12 6.5L8 8V11.5C8 14.5 9.8 17 12 17.8C14.2 17 16 14.5 16 11.5V8L12 6.5Z" fill="white"/>
        </svg>
      </span>
      <span class="logo-text">MapSafe</span>
    </div>
    <nav class="nav-menu">
      <router-link to="/" class="nav-item">
        <span class="nav-icon">🗺️</span> 지도 서비스
      </router-link>
      <router-link to="/photo_analysis" class="nav-item">
        <span class="nav-icon">📷</span> 사진 분석
      </router-link>
    </nav>
  </header>
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
.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 48px;
  background-color: #ffffff;
  border-bottom: 1px solid #e2e8f0;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
  position: sticky;
  top: 0;
  z-index: 100;
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
}

.logo-icon {
  display: flex;
  align-items: center;
}

.logo-text {
  font-size: 20px;
  font-weight: 700;
  color: #1e293b;
}

.nav-menu {
  display: flex;
  gap: 8px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  border-radius: 8px;
  text-decoration: none;
  color: #475569;
  font-weight: 500;
  font-size: 15px;
  transition: all 0.15s ease;
}

.nav-item:hover {
  background-color: #f1f5f9;
  color: #1e293b;
}

.nav-item.router-link-exact-active {
  background-color: #eff6ff;
  color: #2563eb;
}

.nav-icon {
  font-size: 16px;
}
</style>