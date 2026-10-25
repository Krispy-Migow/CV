"""Especie (perro o gato) a partir de la raza.

Oxford-IIIT Pet tiene 37 razas: 12 son de gato y 25 de perro. Las mayúsculas
no sirven para distinguirlas (hay razas de perro con mayúscula, p. ej.
"American Bulldog"), así que usamos la lista explícita.
"""

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair", "Egyptian Mau",
    "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx",
}


def species_of(breed: str) -> str:
    if breed not in CAT_BREEDS:
        return "perro"
    return "gato"
