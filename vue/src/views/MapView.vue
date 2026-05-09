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

      kakao.maps.event.addListener(map, "click", function (mouseEvent) {
        const latlng = mouseEvent.latLng;

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

        roadviewClient.getNearestPanoId(latlng, 50, function (panoId) {
          if (panoId === null) {
            infowindow = new kakao.maps.InfoWindow({
              content: '<div style="padding:8px;">근처에 로드뷰가 없습니다.</div>',
            });

            infowindow.open(map, marker);
            return;
          }

          roadview.setPanoId(panoId, latlng);

          infowindow = new kakao.maps.InfoWindow({
            content: `
              <div style="padding:8px;">
                선택한 위치<br>
                위도: ${latlng.getLat().toFixed(6)}<br>
                경도: ${latlng.getLng().toFixed(6)}
              </div>
            `,
          });

          kakao.maps.event.addListener(marker, "click", function () {
            infowindow.open(map, marker);
          });
        });
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
          return new kakao.maps.LatLng(coord[1], coord[0]);
        });

        const polyline = new kakao.maps.Polyline({
          path,
          strokeWeight: 4,
          strokeColor: "#00AAFF",
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
#mapWrapper {
  position: relative;
  width: 750px;
  height: 350px;
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

#roadview {
  width: 750px;
  height: 350px;
  margin-top: 16px;
}
</style>