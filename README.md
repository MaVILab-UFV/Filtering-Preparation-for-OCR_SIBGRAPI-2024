# A Filtering and Image Preparation Approach to Enhance OCR for Fiscal Receipts

Repository for the paper "A Filtering and Image Preparation Approach to Enhance OCR for Fiscal Receipts" published at at 37th Conference on Graphics, Patterns and Images (SIBGRAPI), 2024.

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
