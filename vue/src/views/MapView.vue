<template>
   <div id="mapContainer">
    <div id="mapWrapper">
      <div id="map"></div>
      <button id="walkBtn" @click="toggleWalkRoads">{{ isWalkVisible ? "도보 끄기" : "도보" }}</button>
    </div>
    <div id="roadview"></div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      map: null,
      walkPolylines: [],
      isWalkVisible: false,
    };
  },

  mounted() {
    if (window.kakao && window.kakao.maps) {
      this.initMap();
    } else {
      const script = document.createElement("script");
      const KAKAO_KEY = import.meta.env.VITE_KAKAO_MAP_KEY;

      script.onload = () => window.kakao.maps.load(this.initMap);
      script.src = `https://dapi.kakao.com/v2/maps/sdk.js?autoload=false&appkey=${KAKAO_KEY}`;

      document.head.appendChild(script);
    }
  },

  methods: {
    initMap() {
      const mapContainer = document.getElementById("map");
      const roadviewContainer = document.getElementById("roadview");

      const mapOption = {
        center: new kakao.maps.LatLng(37.589372, 127.016745),
        level: 3,
      };

      const map = new kakao.maps.Map(mapContainer, mapOption);
      this.map = map;
      const roadview = new kakao.maps.Roadview(roadviewContainer);
      const roadviewClient = new kakao.maps.RoadviewClient();

      let marker = null;
      let infowindow = null;

      kakao.maps.event.addListener(map, "click", async (mouseEvent) => {
        const latlng = mouseEvent.latLng;
        const lat = latlng.getLat();
        const lng = latlng.getLng();

        if (marker) {
          marker.setMap(null);
        }

        if (infowindow) {
          infowindow.close();
        }

        marker = new kakao.maps.Marker({
          position: latlng,
          map: map,
        });

        infowindow = new kakao.maps.InfoWindow({
          content: `
            <div style="padding:10px; width:220px;">
              <b>분석 중...</b><br>
              위도: ${lat.toFixed(6)}<br>
              경도: ${lng.toFixed(6)}
            </div>
          `,
        });

        infowindow.open(map, marker);

        roadviewClient.getNearestPanoId(latlng, 50, function (panoId) {
          if (panoId !== null) {
            roadview.setPanoId(panoId, latlng);
          }
        });

        try {
          const response = await fetch(
            `http://127.0.0.1:8000/predict?lat=${lat}&lng=${lng}&heading=0`
          );

          const result = await response.json();
          const imageUrl = result.image_url;

          if (result.error) {
            infowindow.setContent(`
              <div style="padding:10px; width:220px;">
                <b>분석 실패</b><br>
                ${result.error}
              </div>
            `);
            return;
          }

          // XAI 분석 결과가 포함된 향상된 인포윈도우
          infowindow.setContent(`
            <div style="padding:12px; width:290px; font-family: sans-serif; font-size:13px; line-height: 1.5;">
              <b style="font-size:14px; color:#2c3e50;">📍 선택한 위치 분석</b><br>
              <span style="color:#7f8c8d; font-size:11px;">위도: ${lat.toFixed(5)} / 경도: ${lng.toFixed(5)}</span>
              <hr style="border:none; border-top:1px solid #eee; margin:8px 0;">
              <b>안전 점수:</b> <span style="font-size:15px; font-weight:bold; color:#2e7d32;">${result.safety_score.toFixed(2)}점</span>
              <img
                src="${imageUrl}"
                style="
                  width:220px;
                  height:120px;
                  object-fit:cover;
                  border-radius:8px;
                  margin-top:8px;
                  margin-bottom:8px;
                "
              />
              <div style="margin-top:10px; padding:8px; background:#f8f9fa; border-radius:4px; font-size:12px; color:#455a64; border-left:3px solid #0288d1;">
                ${result.explanation}
              </div>
          `);

        } catch (error) {
          console.error(error);

          infowindow.setContent(`
            <div style="padding:10px; width:220px;">
              <b>서버 연결 오류</b><br>
              FastAPI 서버를 확인하세요.
            </div>
          `);
        }
      });
    },

    async toggleWalkRoads() {
      if (this.isWalkVisible) {
        const currentCenter = this.map.getCenter();
        const currentLevel = this.map.getLevel();

        this.resetMap(currentCenter, currentLevel);

        this.walkPolylines = [];
        this.isWalkVisible = false;
        return;
      }

      await this.drawSafetyRoads();
      this.isWalkVisible = true;
    },

    resetMap(center, level) {
      const mapContainer = document.getElementById("map");
      mapContainer.innerHTML = "";

      const mapOption = {
        center,
        level,
      };

      this.map = new kakao.maps.Map(mapContainer, mapOption);
    },

    async drawSafetyRoads() {
      const response = await fetch("/seongbuk_walk.geojson");
      const geojson = await response.json();

      this.walkPolylines = [];

      geojson.features.forEach((feature) => {
        if (!feature.geometry) return;

        const geometryType = feature.geometry.type;
        const coordinates = feature.geometry.coordinates;

        if (geometryType === "LineString") {
          const path = coordinates.map((coord) => {
            return new kakao.maps.LatLng(coord[0], coord[1]);
          });

          const safeScore = feature.properties?.safeScore ?? 3.0;

          let strokeColor = "#00FF00";
          if (safeScore < 2.5) {
            strokeColor = "#FF0000";
          }
          else if (safeScore < 3.5) {
            strokeColor = "#FFFF00";
          }
          else {
            strokeColor = "#00FF00";
          }

          const polyline = new kakao.maps.Polyline({
            path,
            strokeWeight: 4,
            strokeColor: strokeColor,
            strokeOpacity: 0.8,
            strokeStyle: "solid",
          });

          polyline.setMap(this.map);
          this.walkPolylines.push(polyline);
        }
      });
    },
  },
}
</script>

<style>
#mapContainer {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

#mapWrapper {
  position: relative;
  width: 100%;
  max-width: 1000px;
  height: 600px;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}

#map {
  width: 100%;
  height: 100%;
}

#walkBtn {
  position: absolute;
  top: 12px;
  left: 12px;
  z-index: 9999;

  padding: 8px 14px;
  background-color: white;
  border: 1px solid #999;
  border-radius: 6px;
  cursor: pointer;
  font-weight: bold;
}

#walkBtn:hover {
  background-color: #f0f0f0;
}
</style>