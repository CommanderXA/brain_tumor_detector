import React from "react";

import styles from "./buttons.module.scss";

interface Props extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
  variant?: number;
  fit?: boolean;
}

const SecondaryButton: React.FC<Props> = ({
  children,
  variant = 1,
  fit,
  ...props
}) => {
  return (
    <button
      {...props}
      className={`${
        variant === 1
          ? styles.secondary__variant__1
          : styles.secondary__variant__2
      } ${fit ? "fit" : null}`}
    >
      <h3 style={{margin: 0, padding: 0}}>{children}</h3>
    </button>
  );
};

export default SecondaryButton;
