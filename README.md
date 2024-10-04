# A Filtering and Image Preparation Approach to Enhance OCR for Fiscal Receipts

This repository contains the code for the image pipeline of the paper A Filtering and Image Preparation Approach to Enhance OCR for Fiscal Receipts presented at 37th SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), 2024.

We propose a pipeline divided into two main steps:

1. A filtering approach based on low-level features to identify and discard poor-quality images, select high-quality ones, and flag images that need preparation before OCR; 
2. The flagged images undergo a series of enhancement techniques, including homography transformation, super-resolution, noise reduction, sharpness adjustment, character standardization, and binarization.

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{Auad2024,
    title = {A filtering and image preparation approach to enhance OCR for fiscal receipts},
    author = {Auad, Manoela W and Alves, Sarah C and Kakizaki, Gabriel S and Reis, Julio C. S. and Silva, Michel M},
    year = {2024},
    booktitle = {Proceedings of the 37th Conference on Graphics, Patterns and Images (SIBGRAPI)},
    keywords = {optical character recognition, image filtering, image enhancement, fiscal receipts},
    url = {http://urlib.net/ibi/8JMKD3MGPEW34M/4C22QAL}
}
```
## Instruções de Configuração e Execução

### Pré-requisitos

**Atenção:** Este projeto requer uma GPU para funcionar corretamente.

1. **Python 3.7**: Certifique-se de ter o Python 3.7 instalado na sua máquina.
2. **Tesseract OCR**: Instale o Tesseract OCR. Você pode encontrar as instruções de instalação [aqui](https://github.com/tesseract-ocr/tesseract).
3. **Segment Anythin Model Checkpoint**: Faça download do Model Checkpoint default or vit_h do SAM [aqui](https://github.com/facebookresearch/segment-anything). E salve o arquivo baixado na pasta raiz sibgrapi24-main.

### Passos para Configuração

1. **Criar e ativar o ambiente virtual**:
   - Crie um ambiente virtual chamado `filtering-preparation-ocr` com Python 3.7 usando o comando abaixo:
     ```bash
     python3.7 -m venv filtering-preparation-ocr
     ```
   - Ative o ambiente virtual:
     - **Linux/macOS**:
       ```bash
       source filtering-preparation-ocr/bin/activate
       ```
     - **Windows**:
       ```bash
       filtering-preparation-ocr\Scripts\activate
       ```

2. **Instalar dependências**:
   - Navegue até a pasta `src`:
     ```bash
     cd src
     ```
   - Instale as dependências listadas no arquivo `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

### Executar o Projeto

1. Volte para a raiz do repositório:
   ```bash
   cd ..
    ```

2. Execute o pipeline do Kedro:

   ```bash
   kedro run
    ```

Com esses passos, o ambiente estará configurado e o projeto pronto para ser executado!
