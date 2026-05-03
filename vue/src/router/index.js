import { createRouter, createWebHistory } from "vue-router";
import MapView from "../views/MapView.vue";
import PhotoAnalysisView from "../views/PhotoAnalysisView.vue";

const routes = [
  {
    path: "/",
    name: "map",
    component: MapView,
  },
  {
    path: "/photo_analysis",
    name: "photoAnalysis",
    component: PhotoAnalysisView,
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;
