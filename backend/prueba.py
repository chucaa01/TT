from azure.storage.fileshare import ShareFileClient

# Configuración de tu cuenta
account_name = "dbilletes8263"
account_key = "AtOcMOF+rgfZ83NnBGTR84qHIQKCxpz17Cre2XzTdFvlm1Zg+dSa/U8EDtDrgv468GGb2iDyYarv+ASts5ohxA=="
file_share = "billetes-fotos"
directorio = "imagenes"
archivo_local = r"C:\Users\Jesus\Documents\GitHub\TT\pruebas\prueba_billete.jpg"
nombre_destino = "prueba_billete.jpg"

# Crear cliente del archivo
file_client = ShareFileClient(
    account_url=f"https://{account_name}.file.core.windows.net",
    share_name=file_share,
    file_path=f"{directorio}/{nombre_destino}",
    credential=account_key,
)

# Leer imagen y subirla
with open(archivo_local, "rb") as f:
    content = f.read()

# Crear archivo en Azure y subir contenido
file_client.create_file(size=len(content))
file_client.upload_file(content)

print("✅ Imagen subida correctamente a Azure Files")
print("🔗 URL (no pública, pero válida si tienes acceso):")
print(f"https://{account_name}.file.core.windows.net/{file_share}/{directorio}/{nombre_destino}")
