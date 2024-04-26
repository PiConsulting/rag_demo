import { Example } from "./Example";

import styles from "./Example.module.css";

export type ExampleModel = {
    text: string;
    value: string;
};

const EXAMPLES: ExampleModel[] = [
    {
        text: "¿Las clases de inglés son obligatorias?",
        value: "¿Las clases de inglés son obligatorias?"
    },
    { text: "¿Hasta qué día puedo solicitar mis vacaciones?", value: "¿Hasta qué día puedo solicitar mis vacaciones?" },
    { text: "¿Me podés dar información del programa de referidos?", value: "¿Me podés dar información del programa de referidos?" }
];

interface Props {
    onExampleClicked: (value: string) => void;
}

export const ExampleList = ({ onExampleClicked }: Props) => {
    return (
        <ul className={styles.examplesNavList}>
            {EXAMPLES.map((x, i) => (
                <li key={i}>
                    <Example text={x.text} value={x.value} onClick={onExampleClicked} />
                </li>
            ))}
        </ul>
    );
};
