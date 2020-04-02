## Parameter Configuration:
The following code snippet illustrates the .json structure required to produce valid encryption parameters:
```json
{
  "scheme_name": "",
  "poly_modulus_degree": ,
  "security_level": ,
  "coeff_modulus": [
    
  ],
  "complex_packing":
}
```
### Poly Modulus Degree
The poly modulus degree parameters (poly_modulus_degree) must take on one of the following options: 
**[1024, 2048,  4096, 8192, 16384, 32768]**

### Security Level
The security level (security_level) options are as follows: 
**[0, 128, 192, 256]**

### Coefficient Modulus
The first and last terms in the coefficient modulus must be large to ensure secure encryption and decryption. The intermediate terms should be close together. Microsoft uses the following coefficient modulus:
**[60, 40, 40, 60]**