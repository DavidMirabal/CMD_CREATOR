# CMD CREATOR

## INSTRUCCIONES (SPANISH)

Para que todo funcione correctamente se debe descargar todo este repositorio en su ordenador. A continuación descomprima el .zip descargado y podrá lanzar el programa. Hay dos maneras de lanzar el **CMD_CREATOR**:

* EXCLUSIVO PARA WINDOWS: Ejecutar **CMD_CREATOR_(WINDOWS).exe** y se debería abrir una ventana para introducir algunos parámetros.
* PARA TODOS LOS USUARIOS (ES NECESARIO TENER INSTALADO PYTHON 3 Y ALGUNOS MODULOS TÍPICOS): Ejecutar en un editor o en terminal el archivo **CMD_CREATOR.py**. Debería abrirse una ventana para introducir los parámetros.

En primer lugar, el programa buscará cúmulos globulares en el catálogo de Harris dentro del rango de parámetros que se introduzcan. Se debe introducir una distancia máxima del Sol (en kpc) para la que se quiera buscar cúmulos 
globulares.  En la segunda línea, se introduce la extinción máxima permitida para buscar cúmulos (en E(B-V)). En la tercera línea, se introduce la máxima metalicidad (en [Fe/H]) para la que se quiera buscar cúmulos y de manera
similar en la cuarta línea la mínima metalicidad posible del cúmulo. Al pulsar el botón **Start search** se buscarán en el catálogo de Harris todos los cúmulos que cumplan con los parámetros. Cuando termine de buscar, los 
mostrará en el espacio de abajo. En este paso, se crea automáticamente un archivo .csv con estos cúmulos encontrados y su distancia al Sol, metalicidad y extinción (datos del catálogo de Harris). Este archivo se encuentra en
la carpeta de **Datos/**, en concreto en una carpeta en cuyo nombre se identifican los parámetros de la busqueda de cúmulos. También, se crea una carpeta llamada **Figures.../** donde se guardan las siguientes gráficas de todos 
los cúmulos encontrados:

* CMD: Diagrama color-magnitud con las estrellas filtradas por el método usado en el TFG.
* CMD_ISOCHRONES: Isocronas de edad vieja superpuestas a los CMDs filtrados. Estas isocronas tienen la metalicidad del cúmulo según el catálogo de Harris. Los CMDS están corregidos de extinción.
* GAUSS3D: Gráfica 3D de la gaussiana ajustada al plano del cielo del cúmulo para eliminar las estrellas por crowding (ver TFG).
* PARA-DIST: Distancias de cada estrella del cúmulo al Sol por paralaje y distancia al cúmulo calculada por Baumgardt & Vasiliev (ver TFG).
* XY-GAUSS: Gaussiana ajustada al plano del cielo para eliminar las estrellas por crowding (ver TFG).
* XY-PM: Estrellas del cúmulo en el plano del cielo, se marcan en rojo las eliminadas por crowding. Además, se incluye una figura con los movimientos propios de las estrellas del cúmulo.

Cabe destacar que si en la busqueda de cúmulos se han encontrado 0 o solo 1 cúmulo no se mostrará y se deberá volver a reintentar la busquedad con otros parámetros.

Tras la busqueda, se puede clickar en cualquiera de la tabla donde se muestran los cúmulos filtrados. Al clickar se desplegarán las isocronas superpuestas al CMD filtrado. Si el ajuste de isocrona no es lo suficientemente preciso
es posible ajustar manualmente la extinción y la metalicidad en **Manual fitting**. Si se introducen nuevos valores de extinción y metalicidad, se volverá a mostrar el CMD del cúmulo seleccionado con las isocronas pero con los 
nuevos parámetros introducidos. Esta nueva figura también es guardada en formato .png en **Figures.../CMD-ISOCHRONES(filt)/**. Es de notar que no se tienen todas las isocronas de todas las metalicidades posibles por lo que cuando
se introduzca una metalicidad de la cual no se tiene una isocrona disponible, el programa eligirá automáticamente una isocrona disponible con la metalicidad más cercana. En la parte superior del CMD se muestra la metalicidad
de la isocrona.



## INSTRUCTIONS (ENGLISH)

For everything to work correctly you must download this entire repository to your computer. Then unzip the downloaded .zip file and you will be able to launch the program. There are two ways to launch the **CMD_CREATOR**:

* EXCLUSIVELY FOR WINDOWS: Run **CMD_CREATOR_(WINDOWS).exe** and a window should open to enter some parameters.
* FOR ALL USERS (YOU MUST HAVE PYTHON 3 AND SOME TYPICAL MODULES INSTALLED: NUMPY, MATPLOTLIB, SCIPY AND TKINTER): Run in an editor or terminal the file **CMD_CREATOR.py**. A window should open to enter the parameters.

First, the program will search for globular clusters in the Harris catalog within the range of parameters you enter. You must enter a maximum distance from the Sun (in kpc) for which you want to search for globular clusters. 
In the second line, enter the maximum extinction allowed to search for clusters (in E(B-V)). In the third line, enter the maximum metallicity (in [Fe/H]) for which you want to search for clusters and similarly in the fourth line
the minimum metallicity (in [Fe/H]) for which you want to search for globular clusters. Pressing the **Start search** button will search the Harris catalog for all clusters that meet the parameters. When the search is finished, it will 
will be displayed in the space below. In this step, a .csv file is automatically created with these found clusters and their distance to the Sun, metallicity and extinction (Harris catalog data). This file is located in
the **Datos/** folder, specifically in a folder whose name identifies the parameters of the cluster search. Also, a folder called **Figures.../** is created where the following plots of all the clusters found are stored:

* CMD: Color-magnitude diagram with the stars filtered by the method used in the TFG.
* CMD_ISOCHRONES: Old age isochrones superimposed on the filtered CMDs. These isochrones have the cluster metallicity according to the Harris catalog. The CMDS are extinction corrected.
* GAUSS3D: 3D plot of the Gaussian fitted to the sky plane of the cluster to remove crowding stars (see TFG).
* PARA-DIST: Distances of each cluster star to the Sun by parallax and distance to the cluster calculated by Baumgardt & Vasiliev (see TFG).
* XY-GAUSS: Gaussian adjusted to the plane of the sky to eliminate crowding stars (see TFG).
* XY-PM: Cluster stars in the plane of the sky, the crowded stars are marked in red. In addition, a figure with the proper motions of the cluster stars is included.

It should be noted that if 0 or only 1 cluster has been found in the cluster search, it will not be displayed and the search will have to be reattempted with other parameters.

After the search, you can click on any of the table where the filtered clusters are displayed. Clicking will display the isochrones superimposed on the filtered CMD. If the isochrone adjustment is not precise enough
it is possible to manually fit the extinction and metallicity in **Manual fitting**. If new extinction and metallicity values are entered, the CMD of the selected cluster will be displayed again with the isochrones but with the newly entered parameters. 
new parameters entered. This new figure is also saved in .png format in **Figures.../CMD-ISOCHRONES(filt)/**. It should be noted that not all isochrones of all possible metallicities are available, so when you enter a metallicity for which a
is entered for a metallicity for which no isochrone is available, the program will automatically choose an available isochrone with the closest metallicity. The metallicity of the isochrone is shown at the top of the CMD.
of the isochrone is displayed at the top of the CMD.
