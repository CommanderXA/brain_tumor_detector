import React from "react";

import styles from "./image_viewer.module.scss";
import image from "../../assets/images/image.png";

interface Props {
  image: string | null;
}

const ImageViewer: React.FC<Props> = (props) => {
  return (
    <div
      className={
        props.image !== null
          ? styles.image__container__2
          : styles.image__container
      }
    >
      <img
        className={props.image !== null ? styles.image__2 : styles.image}
        src={props.image !== null ? props.image as string : image}
        alt="Your image"
      />
    </div>
  );
};

export default ImageViewer;
