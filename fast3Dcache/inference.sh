# fast3Dcache
python -m example_f3c --euler_steps 25 \
                      --use_f3c \
                      --image_path /root/autodl-tmp/Fast3Dcache/assets/example_image/typical_building_colorful_cottage.png \
                      --output_name dinosour \
                      --output_dir /root/autodl-tmp/Fast3Dcache/assets/example_image/typical_building_colorful_cottage.png \
                      --full_sampling_ratio 0.2 \
                      --full_sampling_end_ratio 0.75 \
                      --assumed_slope -0.07 \
                      
# vanilla
python -m example_f3c --euler_steps 25 \
                      --image_path /root/autodl-tmp/Fast3Dcache/assets/example_image/typical_building_colorful_cottage.png  \
                      --output_name trellis1 \
