# Cheatsheet de Regular Expressions

Página para simular regular expressions: [RegexPal](https://www.regexpal.com/)

* Disjuncitons: `[a-z]`, `[A-Z]`, `[0123456789]`, etc. Cuando digo `[^e]` significa que busque todo lo que no sea lo que viene después de `^` (es decir, en este caso, la letra `e`). Sin embargo, cuando uso `^` en otro lugar que no sea el comienzo del bracket, significa el caracter `^`.

* Otro tipo de disjunctions: `a|b|c` que es lo mismo que `[abc]`, pero que sirven para buscar secuencias de caracteres. Por ejemplo `[Cc]astor|[Mm]adera` matchea `Castor`, `castor`, `Madera` y `madera`.

* `?`: cero o uno

* `*`: cero o más

* `+`: uno o más

* `.`: cualquier caracter, pero una sola ocurrencia

* Anchors: `^` afuera del bracket significa "principio del string", y `$` afuera del bracket significa fin de linea.

