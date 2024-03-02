## TF CSGO
### TensorFlow Computational Static Graph Oracle
### Visual Studio Code Extension

provides
1. `Decorators`
2. `Intellisense`
3. `Computational Graph Visualization`

This extension tells shape, dtype & device of each tensor according to the following rules:
1. If a user decorates a block and it cannot be deduced, it will trust the user.
2. If a user decorates a block and it's exactly what can be deduced, nothing will happen.
3. If a user decorates a block and it's not what can be deduced, it will `WARN` the user.
