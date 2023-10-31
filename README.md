## Datasets
  - ## Fuji-SfM
    - dataset error
      - "1-Mask-set/training_images_and_annotations/_MG_7954_01.jpg & csv"

  - ## KFuji RGB-DS
    - dataset error
      - "preprocessed data/square_annotations1/BD12_sup_201711_015_05_RGBhr.jpg"

  - ## WSU2019
    - dataset error
      - "CropLoadEstimation/image-33.png" and "CropLoadEstimation/image-33(1).png" is same
      - delete "CropLoadEstimation/image-33(1).png"
      - some image is duplicated but have different labels
        - image-32 == image-33
        - image-181 == image-182
        - image-184 ~= image-185
        - image-196 == image-197
        - image-207 == image-208
        - image-214 == image-215
        - image-228 ~= image-229
        - image-233 ~= image-234
        - image-237 == image-238