# TAIPA Setup Instructions

## Configuração do Ambiente Virtual

1. **Ativar o ambiente virtual:**
   ```bash
   ./activate.sh
   ```
   Ou manualmente:
   ```bash
   source venv/bin/activate
   ```

2. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Executar o aplicativo:**
   ```bash
   streamlit run app.py
   ```

## Desativar o ambiente virtual
```bash
deactivate
```

## Notas
- Sempre use o ambiente virtual para evitar conflitos de dependências
- O arquivo .gitignore já está configurado para ignorar o venv
- As versões das dependências estão fixadas para garantir compatibilidade