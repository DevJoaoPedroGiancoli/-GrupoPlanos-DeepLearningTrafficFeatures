
class MinhaClasse:
    def __init__(self, nome, numero):
        self.nome = nome
        self.numero = numero

# Criar um objeto da classe MinhaClasse
objeto1 = MinhaClasse("Objeto 1", 42)

# Criar variáveis fora da classe
outra_variavel = "Isso não é um atributo de instância da classe"
outra_numero = 100

# Acessar atributos do objeto
print(objeto1.nome)  # Imprime "Objeto 1"
print(objeto1.numero)  # Imprime 42

# Acessar variáveis fora da classe
print(outra_variavel)  # Imprime "Isso não é um atributo de instância da classe"
print(outra_numero)  # Imprime 100