#!/bin/bash
# i don't think this magic worked :(
export SAT_DIR="$PROJECT_HOME/viirs/tifs"
export GRID_DIR="$PROJECT_HOME/gridmet/gridmet-data"
export OUT_DIR="$PROJECT_HOME/gridmet/gridmet-data-cropped"

# for f in $(ls $SAT_DIR | grep NDVI); do
#     echo $f
# done

cat <<EOF > crop_area.geojson
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "Polygon",
      "coordinates": [[
        [-96.6125, 43.5041667],
        [-96.6125, 40.2666667],
        [-89.9458333, 40.2666667],
        [-89.9458333, 43.5041667],
        [-96.6125, 43.5041667]
      ]]
    },
    "properties": {}
  }]
}
EOF



for f in $(ls $GRID_DIR); do
    echo $f
    gdalwarp -r bilinear -tr .5 .5 -t_srs '+proj=longlat +datum=WGS84 +no_defs' -cutline crop_area.geojson -crop_to_cutline $GRID_DIR/$f $OUT_DIR/$f 
done