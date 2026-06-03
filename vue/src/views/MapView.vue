<template>
  <div class="map-page">
    <h2 class="page-title">안전 지도 서비스</h2>

    <div id="mapContainer">
      <div id="mapWrapper">
        <div id="map"></div>
        <button id="walkBtn" :disabled="isWalkLoading" @click="toggleWalkRoads">
          {{ isWalkLoading ? "처리 중..." : (isWalkVisible ? "도보 끄기" : "도보") }}
        </button>
         <button id="locationBtn" @click="toggleCurrentLocation">
          {{ currentMarker ? "현재 위치 끄기" : "현재 위치" }}
        </button>

        <!-- 안전 수준 범례 -->
        <div class="map-legend">
          <div class="legend-title">안전 수준</div>
          <div class="legend-row">
            <span class="legend-dot" style="background:#ef4444;"></span> 위험
          </div>
          <div class="legend-row">
            <span class="legend-dot" style="background:#eab308;"></span> 보통
          </div>
          <div class="legend-row">
            <span class="legend-dot" style="background:#22c55e;"></span> 안전
          </div>
        </div>
      </div>
      <div id="roadview"></div>
    </div>
  </div>
</template>

<script>
export default {
  data() {
    return {
      map: null,
      walkPolylines: [],
      isWalkVisible: false,
      isWalkLoading: false,
      marker: null,
      infowindow: null,
      currentLatLng: null,
      nearbyPolylines: [],
      currentMarker: null,
      watchId: null,
      roadview: null,
      roadviewClient: null,
      analysisRequestId: 0,
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

  beforeUnmount() {
    if (this.watchId) {
      navigator.geolocation.clearWatch(this.watchId);
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

      this.map = new kakao.maps.Map(mapContainer, mapOption);
      this.roadview = new kakao.maps.Roadview(roadviewContainer);
      this.roadviewClient = new kakao.maps.RoadviewClient();

      kakao.maps.event.addListener(this.map, "click", (mouseEvent) => {
        if (this.isWalkVisible) {
          return;
        }

        const latlng = mouseEvent.latLng;
        const lat = latlng.getLat();
        const lng = latlng.getLng();

        const fetchUrl = `http://127.0.0.1:8000/predict?lat=${lat}&lng=${lng}&heading=0`;
        this.analyzeLocation(latlng, fetchUrl);
      });
    },

    startCurrentLocationTracking() {
      if (!navigator.geolocation) {
        alert("이 브라우저는 위치 정보를 지원하지 않습니다.");
        return;
      }

      this.watchId = navigator.geolocation.watchPosition(
        (position) => {
          const lat = position.coords.latitude;
          const lng = position.coords.longitude;

          const currentPosition = new kakao.maps.LatLng(lat, lng);
          this.currentLatLng = currentPosition;

          if (!this.currentMarker) {
            const markerImage = new kakao.maps.MarkerImage(
              "https://t1.daumcdn.net/localimg/localimages/07/mapapidoc/markerStar.png",
              new kakao.maps.Size(24, 35)
            );

            this.currentMarker = new kakao.maps.Marker({
              position: currentPosition,
              map: this.map,
              title: "현재 위치",
              image: markerImage,
              zIndex: 9999,
            });
            //테스트 this.map.setCenter(currentPosition);this.map.setLevel(2); 주석
            this.map.setCenter(currentPosition);
            this.map.setLevel(2);
            this.drawNearbyRoads();
          } else {
            this.currentMarker.setPosition(currentPosition);
          }
        },
        (error) => {
          console.error("위치 오류:", error);
          console.log("에러코드:", error.code);
          console.log("에러메시지:", error.message);

          if (error.code === error.PERMISSION_DENIED) {
            alert("위치 권한이 거부되었습니다. 브라우저 설정에서 위치 권한을 허용해주세요.");
          } else {
            alert("현재 위치를 가져올 수 없습니다.");
          }
        },
        {
          enableHighAccuracy: true,
          maximumAge: 1000,
          timeout: 10000,
        }
      );
    },

    toggleCurrentLocation() {
      if (this.currentMarker) {
        this.currentMarker.setMap(null);
        this.currentMarker = null;

        if (this.watchId) {
          navigator.geolocation.clearWatch(this.watchId);
          this.watchId = null;
        }
        
        this.nearbyPolylines.forEach(polyline => {
          polyline.setMap(null);
        });
        this.currentLatLng = null;
        this.nearbyPolylines = [];
        return;
      }

      this.startCurrentLocationTracking();
    },

    async analyzeLocation(latlng, fetchUrl) {
      const requestId = ++this.analysisRequestId;
      const lat = latlng.getLat();
      const lng = latlng.getLng();

      if (this.marker) {
        this.marker.setMap(null);
        this.marker = null;
      }

      if (this.infowindow) {
        this.infowindow.close();
        this.infowindow = null;
      }

      this.marker = new kakao.maps.Marker({
        position: latlng,
        map: this.map,
      });
      const currentMarker = this.marker;

      kakao.maps.event.addListener(currentMarker, "click", () => {
        kakao.maps.event.preventMap();
        if (this.marker === currentMarker) {
          this.closeAnalysis();
        }
      });

      this.infowindow = new kakao.maps.InfoWindow({
        content: `
          <div style="padding:10px; width:220px;">
            <b>분석 중...</b><br>
            위도: ${lat.toFixed(6)}<br>
            경도: ${lng.toFixed(6)}
          </div>
        `,
      });

      this.infowindow.open(this.map, this.marker);

      this.roadviewClient.getNearestPanoId(latlng, 50, (panoId) => {
        if (panoId !== null) {
          this.roadview.setPanoId(panoId, latlng);
        }
      });

      try {
        const response = await fetch(fetchUrl);
        const result = await response.json();
        const imageUrl = result.image_url;
        const analysisBasis = this.getAnalysisBasisLabel(result.analysis_basis);
        const analysisTitle = result.analysis_basis === "cached_segment_4dir_average"
          ? "선택한 도로 구간 분석"
          : "선택한 위치 분석";

        if (requestId !== this.analysisRequestId) {
          return;
        }

        if (result.error) {
          this.infowindow.setContent(`
            <div style="padding:10px; width:220px;">
              <b>분석 실패</b><br>
              ${result.error}
            </div>
          `);
          return;
        }

        this.infowindow.setContent(`
          <div style="padding:12px; width:290px; font-family: sans-serif; font-size:13px; line-height: 1.5;">
            <b style="font-size:14px; color:#2c3e50;">📍 ${analysisTitle}</b><br>
            <span style="color:#7f8c8d; font-size:11px;">위도: ${lat.toFixed(5)} / 경도: ${lng.toFixed(5)}</span>
            <br><span style="color:#7f8c8d; font-size:11px;">분석 기준: ${analysisBasis}</span>
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
          </div>
        `);
      } catch (error) {
        if (requestId !== this.analysisRequestId) {
          return;
        }

        console.error(error);
        this.infowindow.setContent(`
          <div style="padding:10px; width:220px;">
            <b>서버 연결 오류</b><br>
            FastAPI 서버를 확인하세요.
          </div>
        `);
      }
    },

    closeAnalysis() {
      this.analysisRequestId += 1;

      if (this.marker) {
        this.marker.setMap(null);
        this.marker = null;
      }

      if (this.infowindow) {
        this.infowindow.close();
        this.infowindow = null;
      }
    },

    async toggleWalkRoads() {
      if (this.isWalkLoading) {
        return;
      }

      this.isWalkLoading = true;
      if (this.isWalkVisible) {
        try {
          this.walkPolylines.forEach(polyline => polyline.setMap(null));
          this.walkPolylines = [];
          this.isWalkVisible = false;
          return;
        } finally {
          this.isWalkLoading = false;
        }
      }

      try {
        await this.drawSafetyRoads();
        this.isWalkVisible = true;
      } finally {
        this.isWalkLoading = false;
      }
    },

    async drawSafetyRoads() {
      const response = await fetch("/seongbuk_walk.geojson");
      const geojson = await response.json();

      let apiData = [];
      try {
        const apiResponse = await fetch("http://127.0.0.1:8000/api/safety/all");
        if (!apiResponse.ok) {
          throw new Error(`캐싱 데이터 API 오류: ${apiResponse.status}`);
        }
        apiData = await apiResponse.json();
      } catch (err) {
        console.error("캐싱 데이터 로드 실패", err);
      }

      this.walkPolylines = [];

      const grayPaths = [];
      geojson.features.forEach((feature) => {
        if (!feature.geometry) return;

        const geometryType = feature.geometry.type;
        const coordinates = feature.geometry.coordinates;

        if (geometryType === "LineString") {
          const path = coordinates.map((coord) => {
            return new kakao.maps.LatLng(coord[0], coord[1]);
          });

          grayPaths.push(path);
        }
      });

      this.addWalkPolyline(grayPaths, "#CCCCCC");

      const analyzedRoadsByColor = {};
      apiData.forEach((item) => {
        const coordinates = [
          item.start_latitude,
          item.start_longitude,
          item.end_latitude,
          item.end_longitude,
          item.predicted_score,
        ];

        if (coordinates.some((value) => value == null || !Number.isFinite(Number(value)))) {
          return;
        }

        const path = [
          new kakao.maps.LatLng(item.start_latitude, item.start_longitude),
          new kakao.maps.LatLng(item.end_latitude, item.end_longitude),
        ];

        const strokeColor = this.getScoreColor(item.predicted_score);
        if (!analyzedRoadsByColor[strokeColor]) {
          analyzedRoadsByColor[strokeColor] = [];
        }
        analyzedRoadsByColor[strokeColor].push({ item, path });
      });

      Object.entries(analyzedRoadsByColor).forEach(([strokeColor, roads]) => {
        this.addWalkPolyline(
          roads.map((road) => road.path),
          strokeColor,
          (latlng) => {
            const observation = this.findNearestObservation(latlng, roads);
            return `http://127.0.0.1:8000/api/safety/observations/${observation.id}`;
          },
          2,
        );
      });
    },

    async drawNearbyRoads() {
      //테스트 if(!this.currentLatLng) 주석처리
      if (!this.currentLatLng) {
        console.warn("현재 위치가 아직 없습니다.");
        return;
      }
      //테스트 성북구청 위치: 37.589372, 127.016745

      const lat = this.currentLatLng.getLat();
      const lng = this.currentLatLng.getLng();

      let nearbyData = [];

      try {
        const response = await fetch(
          `http://127.0.0.1:8000/api/safety/nearby?lat=${lat}&lng=${lng}&radius=80` //반경 설정
        );

        if (!response.ok) {
          throw new Error(`nearby API 오류: ${response.status}`);
        }

        nearbyData = await response.json();
        console.log("nearbyData 개수:", nearbyData.length);
        console.log("nearbyData:", nearbyData);
      } catch (error) {
        console.error("주변 안전 데이터 로드 실패:", error);
        return;
      }

      // 기존 주변 색칠 polyline만 제거
      this.nearbyPolylines.forEach(polyline => {
        polyline.setMap(null);
      });
      this.nearbyPolylines = [];
      nearbyData.forEach((item) => {
        if (
          item.start_latitude == null ||
          item.start_longitude == null ||
          item.end_latitude == null ||
          item.end_longitude == null
        ) {
          return;
        }

        const path = [
          new kakao.maps.LatLng(item.start_latitude, item.start_longitude),
          new kakao.maps.LatLng(item.end_latitude, item.end_longitude),
        ];

        const polyline = new kakao.maps.Polyline({
          map: this.map,
          path,
          strokeWeight: 6,
          strokeColor: this.getScoreColor(item.predicted_score),
          strokeOpacity: 0.9,
          strokeStyle: "solid",
          zIndex: 10,
        });

        this.nearbyPolylines.push(polyline);
      });
    },

    getScoreColor(predictedScore) {
      if (predictedScore < 2.5) {
        return "#FF0000";
      }
      if (predictedScore < 3.5) {
        return "#FFFF00";
      }
      return "#00FF00";
    },

    getAnalysisBasisLabel(analysisBasis) {
      if (analysisBasis === "cached_segment_4dir_average") {
        return "저장된 도로 구간의 대표 안전도";
      }
      if (analysisBasis === "realtime_single_heading") {
        return "클릭 위치의 단일 방향 실시간 분석";
      }
      return "이미지 분석";
    },

    findNearestObservation(latlng, roads) {
      const latitude = latlng.getLat();
      const longitude = latlng.getLng();
      let nearestRoad = roads[0];
      let nearestDistance = Infinity;

      roads.forEach((road) => {
        const item = road.item;
        const distance = this.getSquaredDistanceToSegment(
          latitude,
          longitude,
          Number(item.start_latitude),
          Number(item.start_longitude),
          Number(item.end_latitude),
          Number(item.end_longitude),
        );
        if (distance < nearestDistance) {
          nearestRoad = road;
          nearestDistance = distance;
        }
      });

      return nearestRoad.item;
    },

    getSquaredDistanceToSegment(pointLat, pointLng, startLat, startLng, endLat, endLng) {
      const latDelta = endLat - startLat;
      const lngDelta = endLng - startLng;
      const segmentLength = latDelta * latDelta + lngDelta * lngDelta;
      if (segmentLength === 0) {
        return (pointLat - startLat) ** 2 + (pointLng - startLng) ** 2;
      }

      const ratio = Math.max(0, Math.min(1,
        ((pointLat - startLat) * latDelta + (pointLng - startLng) * lngDelta) / segmentLength,
      ));
      const nearestLat = startLat + ratio * latDelta;
      const nearestLng = startLng + ratio * lngDelta;
      return (pointLat - nearestLat) ** 2 + (pointLng - nearestLng) ** 2;
    },

    addWalkPolyline(path, strokeColor, getFetchUrl = null, zIndex = 1) {
      const polyline = new kakao.maps.Polyline({
        path,
        strokeWeight: 4,
        strokeColor,
        strokeOpacity: 0.8,
        strokeStyle: "solid",
        zIndex,
      });

      if (getFetchUrl) {
        kakao.maps.event.addListener(polyline, "click", (mouseEvent) => {
          kakao.maps.event.preventMap();
          const latlng = mouseEvent.latLng;
          this.analyzeLocation(latlng, getFetchUrl(latlng));
        });
      }

      polyline.setMap(this.map);
      this.walkPolylines.push(polyline);
    },
  },
}
</script>

<style>
.map-page {
  display: flex;
  flex-direction: column;
}

.page-title {
  font-size: 24px;
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 20px;
}

#mapContainer {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

#mapWrapper {
  position: relative;
  width: 100%;
  max-width: 1300px;
  height: 600px;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
}

#map {
  width: 100%;
  height: 100%;
}

#walkBtn {
  position: absolute;
  top: 16px;
  left: 16px;
  z-index: 9999;

  padding: 9px 16px;
  background-color: white;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
  cursor: pointer;
  font-weight: 600;
  font-size: 14px;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  transition: all 0.15s ease;
}

#walkBtn:hover {
  background-color: #f8fafc;
}

#walkBtn:disabled {
  cursor: wait;
  opacity: 0.7;
}

#locationBtn {
  position: absolute;
  top: 54px;
  left: 12px;
  z-index: 9999;

  padding: 8px 14px;
  background-color: white;
  border: 1px solid #999;
  border-radius: 6px;
  cursor: pointer;
  font-weight: bold;
}

#locationBtn:hover {
  background-color: #f0f0f0;
}

/* 안전 수준 범례 */
.map-legend {
  position: absolute;
  top: 16px;
  right: 16px;
  z-index: 9999;
  background: white;
  padding: 12px 16px;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.12);
  font-size: 13px;
}

.legend-title {
  font-weight: 700;
  color: #1e293b;
  margin-bottom: 8px;
}

.legend-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 5px;
  color: #475569;
}

.legend-dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
  display: inline-block;
}

#roadview {
  width: 100%;
  max-width: 1300px;
  height: 350px;
  margin-top: 20px;
  border-radius: 12px;
  overflow: hidden;
}
</style>
