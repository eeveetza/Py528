# Python Implementation of Recommendation ITU-R P.528

This code repository contains a python software implementation of  [Recommendation ITU-R P.528-5](https://www.itu.int/rec/R-REC-P.528/en) with a propagation prediction method for aeronautical mobile and radionavigation services in the frequency range 100 - 30000 MHz.  

This is a translation of the original reference C++ implementation of this Recommendation available at [NTIA/p528](https://github.com/NTIA/p528) provided by the US National Telecommunications and Information Administration [NTIA](https://www.ntia.gov). This version of the code corresponds to the reference MATLAB/Octave version approved by ITU-R Working Party 3K and published by Study Group 3 on [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/rwp3m/Pages/digprod.aspx).


The package can be downloaded and installed using:
~~~
python -m pip install "git+https://github.com/eeveetza/Py528/#egg=Py528"   
~~~

and imported as follows
~~~
from Py528 import P528
~~~

The following table describes the structure of the folder.

| File/Folder               | Description                                                         |
|----------------------------|---------------------------------------------------------------------|
|`/src/Py528/P528.py`                | python implementation of Recommendation ITU-R P.528        |
|`/tests/validateP528.py`          | python script used to validate the implementation of Recommendation ITU-R P.528 in `P528.bt_loss()`             |
|`/tests/Data_Tables/`    | Folder containing a set of validation examples for software implementations  of this Recommendation |


## Function Call

~~~
result = P528.bt_loss(d_km, h_1_meter, h_2_meter, f_mhz, T_pol, p)
result = P528.bt_loss(d_km, h_1_meter, h_2_meter, f_mhz, T_pol, p, use_reflection = false)
~~~ 

## Input arguments of function `P528.bt_loss`

| Variable          | Type   | Units | Limits       | Description  |
|-------------------|--------|-------|--------------|--------------|
| `d_km`               | scalar double | km   | 0 < `d`   | Great circle path distance between terminals  |
| `h_1_meter`      | scalar double | m    | 1.5 ≤ `h_1_meter` ≤ 20000 | Height of the low terminal |
| `h_2_meter`      | scalar double | m    | 1.5 ≤ `h_2_meter` ≤ 20000 | Height of the high terminal |
| `f_mhz`          | scalar double | MHz    | 100 ≤ `f_mhz` ≤ 30000   | Frequency|
| `T_pol`           | scalar int    |       |             |  Polarization <br> 0 = horizontal <br> 1 = vertical |
| `p`          | scalar double | %    | 1 ≤ `p` ≤ 99   | Time percentage|
| `use_reflection`          | scalar bool | -    | True (default), False   | A flag that removes the ground reflection (optional)|


 
## Outputs ##

Outputs are contained within a defined `result` structure:

| Variable   | Type   | Units | Description |
|------------|--------|-------|-------------|
| `A_db`    | double | dB    | Basic transmission loss |
| `d_km`	| double  |	km	|Great circle path distance. Could be slightly different than specified in input variable if within LOS region |
| `A_fs_db`    | double | dB    | Free-space basic transmission loss |
| `A_a_db`    | double | dB    | Median atmospheric absorption loss |
| `theta_h1_rad`    | double | rad    | Elevation angle of the ray at the low terminal|
| `propagation_mode`    | int |    | Mode of propagation <br>1 = Line of Sight<br> 2 = Diffraction<br> 3 = Troposcatter|
| `rtn`    | int |    | Return flags / Error codes|ic transmission loss not exceeded Tpc % time |


## Software Versions
The code was tested and runs on:
* python3.9

## References

* [Recommendation ITU-R P.528](https://www.itu.int/rec/R-REC-P.528/en)

* [ITU-R SG 3 Software, Data, and Validation Web Page](https://www.itu.int/en/ITU-R/study-groups/rsg3/rwp3m/Pages/digprod.aspx)

* [MATLAB/Octave Implementation of Recommendation ITU-R P.528](https://github/eeveetza/p528)

* [NTIA/p528](https://github.com/NTIA/p528)
