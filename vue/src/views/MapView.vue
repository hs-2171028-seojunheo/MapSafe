<template>
  <div id="mapContainer">
    <div id="map"></div>
    <div id="roadview"></div>
  </div>
</template>

<script>
export default {
  data() {
    return {};
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
        center: new kakao.maps.LatLng(37.562632898194835, 126.9454282268269),
        level: 3,
      };

      const map = new kakao.maps.Map(mapContainer, mapOption);
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
  },
};
</script>

<style>
#map {
  width: 750px;
  height: 350px;
}

#roadview {
  width: 750px;
  height: 350px;
  margin-top: 16px;
}
</style>