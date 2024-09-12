import pandas as pd
from faker import Faker

# Erstellen Sie ein Faker-Objekt
fake = Faker()

# Lesen Sie die CSV-Datei in einen DataFrame
df = pd.read_csv('sales_messages.csv')

# Erstellen Sie eine neue Spalte "RandomMessage", indem Sie für jede Zeile eine zufällige Nachricht generieren
df['RandomMessage'] = [fake.sentence() for _ in range(len(df))]

# Speichern Sie den DataFrame wieder als CSV-Datei
df.to_csv('sales_messages.csv', index=False)