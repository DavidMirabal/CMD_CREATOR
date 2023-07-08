# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:58:42 2023

@author: mirab
"""

import tkinter as tk
import numpy as np
from tkinter import ttk, NS, Canvas, Scrollbar
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from tkinter.font import Font
import warnings
import matplotlib
import matplotlib
matplotlib.use('agg')

control_table=0
def starter():
    global R_sunMAX
    global E_max
    global Fe_max
    global Fe_min
    global data
    global control_table
    global directorio
    global directorio_isocronas
    global directoriodata
    global table
    global nombre
    
    start.grid_forget()
    #Maximum distance from the Sun to the cluster, in kpc
    R_sunMAX  = float(num_rsun.get())

    #Maximum extinction (E(B-V)) allowed
    E_max = float(num_Emax.get())

    #Maximum metallicity ([Fe/H]) allowed
    Fe_max = float(num_Femax.get())

    #Minimu metallicity ([Fe/H]) allowed
    Fe_min = float(num_Femin.get())
    
    if os.path.exists('.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')'):
        directorio = '.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')'
        directorio_isocronas = '.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')\\CMD_ISOCHRONES'
    else:
        os.makedirs('.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')')
        os.makedirs('.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')\\CMD(filt)')
        os.makedirs('.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')\\PARA_DIST(filt)')
        os.makedirs('.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')\\XY_PM(filt)')
        os.makedirs('.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')\\CMD_ISOCHRONES')
        os.makedirs('.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')\\XY_GAUSS(filt)')
        os.makedirs('.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')\\GAUSS3D(filt)')
        directorio = '.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')'
        directorio_isocronas = '.\\Figures('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')\\CMD_ISOCHRONES'

    if os.path.exists('.\\Datos\\('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')'):
        directoriodata = '.\\Datos\\('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')'
    else:
        os.makedirs('.\\Datos\\('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')')
        directoriodata = '.\\Datos\\('+ str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ')'
    

    
    
    
    clasificacion_cumulos()
    
    
    try:
        warning = tk.Label(fr, text='Wait a moment', font=Font(weight="bold", size=14), fg='red')
        warning.grid(row=1, column=2, sticky='e', rowspan=4)
        root.update()
        graficador()
    except IndexError:
        warning.destroy()
        root.update()
        warning = tk.Label(fr, text='Only 0 or 1 cluster found, retry', font=Font(weight="bold", size=14), fg='red')
        warning.grid(row=1, column=2, sticky='e', rowspan=4)
        root.update()
        root.after(5000, warning.destroy())
        start.grid(row=1, column=2, rowspan=4,pady=4)    
    

    data = np.loadtxt(directoriodata + r'/GCs(' + str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ').csv', delimiter=';', dtype=str)
    
    
    table.delete(*table.get_children())
    
    
    
    yscrollbar.grid(row=0, column=5, sticky=NS)
    for i in range(len(data[:,0])):
        table.insert(
            "",
            tk.END,
            text=data[i,1],
            values=(float(data[i, 2]), float(data[i,3]), float(data[i,4]))
        )
    
    warning.destroy()
    start.grid(row=1, column=2, rowspan=4,pady=4)
    root.update()
    
    
def clasificacion_cumulos():
    d1 = np.loadtxt(r'./Datos/Cúmulos Globulares (Catálogo) I.csv', skiprows=1, delimiter=';')
    d2 = np.loadtxt(r'./Datos/Cúmulos Globulares (Catálogo) II.csv', skiprows=1, delimiter=';')
    nom = np.loadtxt(r'./Datos/Nombres cúmulos ordenados.csv', delimiter=';', dtype=str,  encoding='utf-8-sig')


    R_sun = np.loadtxt(r'./Datos/Cúmulos_Vasiliev_2.csv', delimiter=';', dtype=str)[:,3]  #Distancia al Sol en kpc
    R_sun = np.array(R_sun, dtype=float)
    R_gc = d1[:,1]   #Distancia al centro galactica en kpc


    index = np.array([], dtype=int)
    for i in range(len(R_sun)):
        if R_sun[i] < R_sunMAX and d2[i,1]<E_max and d2[i,0]<Fe_max and d2[i,0]>Fe_min:
            index = np.append(index, i)
       
    lista_nombres = np.array([])
    for i in index:
        lista_nombres = np.append(lista_nombres, nom[i])
    for i in range(len(lista_nombres)):
        lista_nombres[i] = lista_nombres[i].replace(' ', '_')
        
        


    lista_nombresREAL = np.empty(len(lista_nombres), dtype='U32')
    index2 = np.copy(index)
    index = np.empty(len(index2), dtype='U32')


    nombres_archiv = os.listdir(r'./Datos/Globular clusters from EDR3 (Eugene)/catalogues')
    k = 0
    k2 = 0
    for i in range(len(lista_nombres)):
        for j in range(len(nombres_archiv)):
            if lista_nombres[i] in nombres_archiv[j]:
                lista_nombresREAL[k] = nombres_archiv[j].replace('.txt', '')
                index[k2] = index2[k2]
                k = k +1
                break
        k2 = k2 +1
    i = 0
    while i < len(index):
        if index[i] == '':
            index = np.delete(index, i)
        else:
            i = i + 1
    index = np.array(index, dtype=int)
    i = 0
    while i < len(lista_nombresREAL):
        if lista_nombresREAL[i] == '':
            lista_nombresREAL = np.delete(lista_nombresREAL, i)
        else:
            i = i + 1
    cums = np.array([])    
    metal = np.array([])
    E = np.array([])
    for i in index:
        metal = np.append(metal, d2[i,0])
        cums = np.append(cums, R_sun[i])
        E = np.append(E, d2[i,1])

       
    clusters = np.array([index, lista_nombresREAL, cums, metal, E], dtype='U32')
    clusters = np.transpose(clusters)
    np.savetxt(directoriodata + r'/GCs(' + str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ').csv', clusters, delimiter=';', header=('Índice;Nombres;R_Sun<%f;Metalicidad;E(B-V)' %R_sunMAX), fmt="%s")


def graficador():
    global extincion
    global fig
    cat = 3 
    ext = 4 
    mul_ex = 2
    mul_ey = 2
    bn = 100  #bin del ajuste


    data_vasi = np.loadtxt(r'./Datos/Cúmulos_Vasiliev.csv', delimiter=';', dtype=np.dtype('U19'))
    Caracteristicas = np.loadtxt(directoriodata + r'/GCs(' + str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ').csv', delimiter=';', dtype=str)

    metalicidad = os.listdir(r'./Datos/Isocronas BASTI (elegidos)/')
    metalicidad_N = np.zeros(len(metalicidad), dtype=np.dtype('U25'))


    for i in range(len(metalicidad)):
          if 'm' in metalicidad[i]:
              metal_letras = list(metalicidad[i])
              metalicidad_N[i] = '-' +  metal_letras[4] + '.' + metal_letras[5] + metal_letras[6]
          elif 'p' in metalicidad[i]:
              metal_letras = list(metalicidad_N[i])
              metalicidad_N[i] = '+' + metal_letras[4] + '.' + metal_letras[5] + metal_letras[6]
          
          else:
              print('error')
              break
          
    def gauss(X, x_0, y_0, sigma_x, sigma_y, a):
        x,y = X
        f = a*np.e**(-(((x-x_0)**2)/(2*sigma_x**2) + ((y-y_0)**2)/(2*sigma_y**2)))
        return f

    def elipse(x,y,a,b, h, k):
        return ((x-h)/a)**2 + ((y-k)/b)**2
        
    repetidos = np.array([])
    for j in range(len(Caracteristicas[:,0])):
        if 'extincion' in globals():
            del(extincion)
            
        print('----------------------------------------')
        
        for i in range(len(metalicidad_N)):
            if float(Caracteristicas[j,cat])== float(metalicidad_N[i]) and not(j in repetidos):
                print('Datos:')
                nombre = Caracteristicas[j,1]
                distancia = Caracteristicas[j,2]  #en kpc de Vasiliev
                extincion = Caracteristicas[j,ext]   
                repetidos = np.append(repetidos, j)
                indicek = i
                break
            
        if 'extincion' in globals():
            print(indicek)
            print(Caracteristicas[j,cat], metalicidad_N[indicek])
        else: 
            print('NO SE PLOTEA ESTE CÚMULO')
            continue
        
        
        directorio1 = r'Datos/Isocronas BASTI (elegidos)/' + metalicidad[indicek] + '/'
        archivos = os.listdir(directorio1)
        for i in range(len(archivos)):
            os.rename(directorio1 + archivos[i], directorio1 + list(archivos[i])[0] + list(archivos[i])[1] +  list(archivos[i])[2] + list(archivos[i])[3]  +'0 Myr.txt')
        
        archivos = os.listdir(directorio1)
       
        
        EBV = float(extincion)
        distance = float(distancia)*1000 #en pc
        memberprop_limit = 0.90
        print(nombre)
        print('[Fe/H] = ' + metalicidad_N[indicek])
        print('Distancia al cúmulo: ' + distancia + ' [kpc]')
        print('Extinción NED: ' + str(EBV))
        
        directorios = os.listdir(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/')
        for i in range(len(directorios)):
            if nombre in directorios[i]:
                nombre = directorios[i].replace(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/', '')
                
        data = np.loadtxt(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/' + nombre)
        nombre = nombre.replace('.txt', '')
        
        memberprop = data[:,16]
        mp_index = np.zeros(len(memberprop), dtype=bool)
        for i in range(len(memberprop)):
            if memberprop[i]>memberprop_limit:
                mp_index[i] = True 
        #SE ELIMINAN LAS ESTRELLAS MÁS CERCANAS AL CENTRO DEL CÚMULO
        x = data[:,3]*60   #Posición centrado en el cúmulo(X) en arcmin
        y = data[:,4]*60   #Posición centrado en el cúmulo (Y) en arcmin


        ra = data[:,1]  #Ascensión recta
        dec = data[:,2] #Declinación
        
        
        el = memberprop>memberprop_limit
        xg = np.linspace(np.min(ra[el]), np.max(ra[el]), bn)
        yg = np.linspace(np.min(dec[el]), np.max(dec[el]), bn)
        xg, yg = np.meshgrid(xg, yg)
        H, xedges, yedges = np.histogram2d(ra[el], dec[el], bins=bn)
        p = [(np.max(ra[el])-np.min(ra[el]))/2 + np.min(ra[el]),(np.max(dec[el])-np.min(dec[el]))/2 + np.min(dec[el]),0.1,0.1,np.max(H)]
        popt, cov = curve_fit(gauss, (xg.flatten(), yg.flatten()), H.flatten(), p0=p)

         
        descartes_index = np.zeros(len(memberprop), dtype=bool)
        eleccion_index = np.zeros(len(memberprop), dtype=bool)
        
        a = mul_ex*popt[2]
        b = mul_ey*popt[3]
        for i in range(len(memberprop)):
            if elipse(ra[i], dec[i], a,b, popt[0], popt[1])<1:
                descartes_index[i] = True
            elif memberprop[i]>memberprop_limit:
                eleccion_index[i] = True
        
        
        gid = data[:,0] #GAIA ID
        
        ra = data[:,1]  #Ascensión recta
        dec = data[:,2] #Declinación
        
        x = data[:,3]   #Posición centrado en el cúmulo(X) en grados
        y = data[:,4]   #Posición centrado en el cúmulo (Y) en grados
        
        plx = data[:,5] #Paralaje corregido con zero-point offset usando Lindegren
        plxe = data[:,8]#Error en paralaje
        
        pmra = data[:,6]#Movimiento propio en la direción ra
        pmdec = data[:,7]#Movimiento propio en la dirección dec
        pmrae = data[:,9]#Error en mp en ra
        pmdece = data[:,10]#Error en mp en dec
        pmcorr = data[:,11]#Coeficiente de correlación entre los 2 errores de mp
        
        g_mag = data[:,12]#Magnitud en la banda G
        bp_rp = data[:,13]#BP-RP
        Sigma = data[:,14]#Densidad de fuentes (estrellas/arcmin^2)
        qflag = data[:,15]#0/1: 6/5 parametros; 0/2: estrellas que pasaron los filtros/estrellas que no
        memberprop = data[:,16]
        
        # Colour--Teff relation determined from Gaia DR2 data --> Computed using color_teff_GDR2.ipynb
            

        #Corrección de magnitudes
        ebv_array=np.ones(np.size(g_mag))*EBV
        distance_array = np.ones(np.size(g_mag))*distance

        poly = np.array([62.55257114, -975.10845442, 4502.86260828, -8808.88745592, 10494.72444183])
        Teff = np.poly1d(poly)(bp_rp) / 1e4

        # We enter in an iterative process of 10 steps to refine the AG and E_BP_RP computation. 'Cause we do not have intrinsic colour.
        for i in range(10):
            E_BP_RP = (-0.0698 + Teff*(3.837-2.530*Teff)) * EBV # A_BP - A_RP
            AG = (1.4013 + Teff*(3.1406-1.5626*Teff)) * EBV
            Teff = np.poly1d(poly)(bp_rp - E_BP_RP) / 1e4

        ABP = (1.7895 + Teff*(4.2355-2.7071*Teff)) * ebv_array
        ARP = (1.8593 + Teff*(0.3985-0.1771*Teff)) * ebv_array
        
        MG = g_mag - 5*(np.log10(np.ones(np.size(g_mag))*distance_array)) + 5 - AG
        COL_bp_rp = bp_rp - E_BP_RP
        
        
        stars = len(memberprop) #nº de estrellas
        difstars = stars-len(memberprop[mp_index]) #nº de estrellas eliminadas por la probabilidad
        startselec = len(memberprop[eleccion_index]) #nº de estrellas que cumplen los parametros
        #Plots
        #CMD E HISTOGRAMA
        fig = Figure(figsize=(12, 9))
        ax1 = fig.add_subplot(121)
        ax1.plot(COL_bp_rp[eleccion_index], MG[eleccion_index], '.', markersize=1.3, color='black', label='')
        fig.gca().add_patch(plt.Rectangle([0.5, 4.5], 0.5, -1.5, color='b', alpha=0.2))
        ax1.set_xlim(0, 2)
        
        ax1.set_ylim(6,-3)
        
        ax2 = fig.add_subplot(122)
        ax2.plot(COL_bp_rp[eleccion_index], MG[eleccion_index], '.', markersize=1.3, color='black', label='')
        
        ax2.set_xlim(0.5, 1)
        ax2.set_ylim(4.5,3)
        
        for i in range(len(archivos)):
            data_iso = np.loadtxt(directorio1 + archivos[i])
            M_ini_iso = data_iso[:,0]
            M_fin_iso = data_iso[:,1]
            L_iso = data_iso[:,2]
            Te_iso = data_iso[:,3]
            MG_iso = data_iso[:,4]
            bp_iso = data_iso[:,5]
            rp_iso = data_iso[:,6]
            bp_rp_iso = bp_iso - rp_iso
            plt.subplot(121)
            ax1.plot(bp_rp_iso, MG_iso, label=archivos[i].replace('.txt',''))
            plt.subplot(122)
            ax2.plot(bp_rp_iso, MG_iso, label=archivos[i].replace('.txt',''))
        print(directorio1 + archivos[i])
        # plt.subplot(121)
        ax1.set_xlabel('G$_{BP}$-G$_{RP}$',fontsize=16)
        ax1.set_ylabel('M$_G$',fontsize=16)
        fig.suptitle(nombre + ', [Fe/H] = ' + metalicidad_N[indicek] + ', stars with mul_sigma <%.1f' %mul_ex,fontsize=16)
        ax1.legend(fontsize=14)
        
        # plt.subplot(122)
        ax2.set_xlabel('G$_{BP}$-G$_{RP}$',fontsize=16, color='b')
        ax2.set_ylabel('M$_G$',fontsize=16, color='b')
        ax2.legend(fontsize=14)
        ax = plt.gca() # Obtiene el objeto del eje actual
        for spine in ax.spines.values():
            spine.set_edgecolor('b')
        ax.xaxis.set_tick_params(color='blue', labelcolor='blue')
        ax.yaxis.set_tick_params(color='blue', labelcolor='blue')
        plt.tight_layout()

        fig.savefig(directorio_isocronas + '/' + nombre  + '(CMD_ISO).png', dpi=300)
        
        plt.figure(6, figsize=(6, 6))
        plt.suptitle(nombre, fontsize=16)
        plt.plot(0, -10, '.', color='grey')
        plt.plot(0, -10, '.', color='indianred')
        plt.plot(0, -10, '.', color='black')
        
        plt.plot(COL_bp_rp, MG, '.', markersize=0.7, color='grey')
        plt.plot(COL_bp_rp[descartes_index], MG[descartes_index], '.', markersize=0.7, color='indianred')
        plt.plot(COL_bp_rp[eleccion_index], MG[eleccion_index], '.', markersize=1.3, color='black')
        
        plt.legend(('Stars with MP<%.2f' %memberprop_limit,'Stars with mul_sigma<%.1f' %mul_ex, 'Stars with MP>%.2f and \nwith mul_sigma>%.1f' %(memberprop_limit, mul_ex)))
        plt.xlabel('G$_{BP}$-G$_{RP}$',fontsize=16)
        plt.ylabel('M$_G$',fontsize=16)
        plt.xlim(-1, 3.5)
        if np.isnan(min(MG))==True:
            plt.ylim(8,-5)
        else:
            plt.ylim(8,min(MG))
        plt.tight_layout()
        
        plt.savefig(directorio + '/CMD(filt)/' + nombre + '_hist(%.2f__ms%.1f)).png' %(memberprop_limit, mul_ex), dpi=400)
        plt.close()
        
        plxe = plxe/1000
        plx = plx/1000
        pesos = abs(1/(plx[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5])**2)*plxe[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5]
        
        pesos = (1/pesos)**2
        print(1/np.sqrt(np.sum(pesos)))
        print('-----------------------------')
        print(nombre + ': Se han eliminado %d estrellas, es decir un %.1f %s de las estrellas totales' %(difstars, 100*difstars/len(memberprop), '%'))
        # print('Distance by Gaia parallax = ' + str(round(np.mean(abs(1000/plx[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5])),3)) + ' $\pm$ ' + str(round(np.std(abs(1000/plx[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5])),3)))
        
        print('-----------------------------')
        
        plxe = plxe*1000
        plx = plx*1000
        #POSICION Y MOVIMIENTOS PROPIOS
        plt.figure(2, figsize=(12, 6))
        ax1 = plt.subplot(121)

        plt.suptitle(nombre, fontsize=14)

        plt.plot(ra, dec, '.', markersize=0.7, color='grey', label=('Stars with MP<%.2f' %memberprop_limit))
        
        plt.plot(ra[descartes_index], dec[descartes_index], '.', markersize = 0.7, color='indianred', label = ('Stars with mul_sgima<%.1f' %mul_ex))
        plt.plot(ra[eleccion_index], dec[eleccion_index], '.', markersize=1.3, color='black', label = ('Stars with MP>%.2f' %memberprop_limit))
        plt.legend()
        plt.xlabel('ra [deg]',fontsize=16)
        plt.ylabel('dec [deg]',fontsize=16)

        ax2 = plt.subplot(122)

        ax2.plot(pmra, pmdec, '.', color='gray', markersize=0.7)
        ax2.plot(pmra[descartes_index], pmdec[descartes_index], '.', color='indianred', markersize=0.7)
        ax2.plot(pmra[eleccion_index], pmdec[eleccion_index], '.', color='black', markersize=1.3)

        #plt.errorbar(pmra, pmdec, xerr=pmrae, yerr=pmdece, marker='.', color='gray', markersize=0.7)
        #plt.errorbar(pmra[mp_index], pmdec[mp_index], xerr=pmrae[mp_index], yerr=pmdece[mp_index], marker='.', color='black', markersize=1.3)
        ax2.legend(('Stars with MP<%.2f' %memberprop_limit, 'Stars with mul_sigma<%.1f' %mul_ex, 'Stars with MP>%.2f' %memberprop_limit))
        plt.xlabel('Proper Motion ra [mas/yr]', fontsize=16)
        plt.ylabel('Proper Motion dec [mas/yr]', fontsize=16)
        plt.tight_layout()
        plt.savefig(directorio + '/XY_PM(filt)/' + nombre + '_XY_PM(%.2f__ms%.1f)).png' %(memberprop_limit, mul_ex), dpi=400)
        plt.close()
        
        plt.figure(3, figsize=(7.5, 6))
        ax = plt.subplot()
        plt.suptitle(nombre, fontsize=16)
        gaussiana = gauss((xg,yg), popt[0], popt[1], popt[2],popt[3], popt[4])
        plt.contourf(xg, yg, gaussiana, 15)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize='large')
        cbar.set_label(label='Fit stars number',size=16)
        plt.plot(ra[eleccion_index], dec[eleccion_index], '.', markersize=1.3, color='black', label='Selected stars')
        elipse_plot = Ellipse((popt[0], popt[1]), 2*a, 2*b, ls='-', color='r', lw=1, fill=False, label='Cutting ellipse')
        ax.add_patch(elipse_plot)
        elipse_plot.set_zorder(5)
        plt.legend(fontsize=16)
        plt.xlabel('ra [deg]',fontsize=16)
        plt.ylabel('dec [deg]',fontsize=16)
        plt.savefig(directorio + '/XY_GAUSS(filt)/' + nombre + '_XY_GAUSS(%.2f__ms%.1f)).png' %(memberprop_limit, mul_ex), dpi=400)
        plt.close()
        
        
        plt.figure(4, figsize=(6, 6))
        plt.suptitle(nombre, fontsize=16)
        ax2 = plt.subplot(111,projection = '3d')
        ax2.plot_surface(xg, yg, gaussiana)
        ax2.plot(ra[eleccion_index], dec[eleccion_index], '.', markersize=1.3, color='black')
        plt.xlabel('ra [deg]',fontsize=16)
        plt.ylabel('dec [deg]',fontsize=16)
        ax2.set_zlabel('Fit stars number' , fontsize=16)
        ax2.plot(popt[0] + a*np.cos(np.linspace(0,2*np.pi,100)), popt[1] + b*np.sin(np.linspace(0,2*np.pi,100)), np.zeros(100), zorder=6, label='Cutting ellipse')
        plt.legend()
        plt.savefig(directorio + '/GAUSS3D(filt)/' + nombre + '_GAUSS3D(%.2f__ms%.1f)).png' %(memberprop_limit, mul_ex), dpi=400)
        plt.close()
        
        
        
        
        plt.figure(5)
        plt.suptitle(nombre, fontsize=14)
        plt.plot(abs(1000/plx[eleccion_index]), abs(plx[eleccion_index]/plxe[eleccion_index]), '.', color='black', markersize=1.3, label='Stars')
        plt.xlim(distance-1*distance, distance*1+distance)
        plt.axvline(distance, ls='-', color='magenta', label='Distance calculated by \nBaumgardt & Vasiliev (2021)')
        try:   
            plt.axvline(np.average(abs(1000/plx[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5]), weights=pesos), ls='-', color='orange', label='Distance calculated from \nGaia parallaxes')
            plt.axvline(np.average(abs(1000/plx[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5]), weights=pesos) + np.std(abs(1000/plx[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5])), ls='--', color='orange', alpha=0.5, label='Distance calculated from \nGaia parallaxes \n$\pm$ 1 standar deviation')
            plt.axvline(np.average(abs(1000/plx[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5]), weights=pesos) - np.std(abs(1000/plx[eleccion_index][plx[eleccion_index]/plxe[eleccion_index] > 5])), ls='--', color='orange', alpha=0.5)
        except:
            continue
        plt.xlabel('Star Distance [pc]', fontsize=12)
        plt.ylabel('Parallax over parallax error' , fontsize=12)
        plt.axhspan(0, 5, color='grey', alpha=0.5, label='Excluded stars')
        plt.legend()
        plt.savefig(directorio + '/PARA_DIST(filt)/' + nombre + '_PARA_DIST(filt)(%.2f__ms%.1f)).png' %(memberprop_limit, mul_ex), dpi=400)
        plt.tight_layout()
        
        j = j+1
        del(extincion)
        plt.close()

root = tk.Tk()
icon = tk.PhotoImage(file="icon.png")

root.iconphoto(True, icon)
root.title('GCs: CMD and Isochrones')


fr = ttk.Frame(root)
fr.pack(side="left", fill="both", anchor='n', expand=True)

fr2 = ttk.Frame(root)
fr2.pack(side="bottom", fill="both", expand=True)
#----------------Initial Values-----------------------------------------

label_title = tk.Label(fr, text='Values to search for globular clusters', font=Font(weight="bold", size=14))
label_title.grid(row=0, column=0, sticky='e', columnspan=3,pady=4)

rsun = tk.Label(fr, text='Max. distance to Sun [kpc] =')
rsun.grid(row=1, column=0, sticky='e')

num_rsun = tk.StringVar()
rsun_entry = tk.Entry(fr, textvariable=num_rsun)
rsun_entry.grid(row=1, column=1)


E_max = tk.Label(fr, text='Max. E(B-V) =')
E_max.grid(row=2, column=0, sticky='e')

num_Emax = tk.StringVar()
Emax_entry = tk.Entry(fr, textvariable=num_Emax)
Emax_entry.grid(row=2, column=1)


Fe_max = tk.Label(fr, text='Max. [Fe/H] =')
Fe_max.grid(row=3, column=0, sticky='e')

num_Femax = tk.StringVar()
Femax_entry = tk.Entry(fr, textvariable=num_Femax)
Femax_entry.grid(row=3, column=1)


Fe_min = tk.Label(fr, text='Min. [Fe/H] =')
Fe_min.grid(row=4, column=0, sticky='e')

num_Femin = tk.StringVar()
Femin_entry = tk.Entry(fr, textvariable=num_Femin)
Femin_entry.grid(row=4, column=1)


#---------------------Button start--------------------------------------------

icon_search = tk.PhotoImage(file='icon_search.png').subsample(-5, 5)
start = tk.Button(fr, text='Start search', width=15, command=starter, image=icon_search, compound=tk.TOP)
start.grid(row=1, column=2, rowspan=4,pady=4)
start.config(width=60, height=60)

#------------------Table-----------------------------------------------------

canvas = tk.Canvas(fr)
yscrollbar = ttk.Scrollbar(canvas, orient="vertical")
table = ttk.Treeview(canvas, columns=('Distance to the Sun',  '[Fe/H]', 'E(B-V)'), yscrollcommand=yscrollbar.set)
table.grid(row=0, column=0, columnspan=4, pady=4, padx=4)
yscrollbar.configure(command=table.yview)
table.heading("#0", text="GC name")
table.heading('Distance to the Sun', text='Distance to the Sun [kpc]')
table.heading('E(B-V)', text='E(B-V)')
table.heading('[Fe/H]', text='[Fe/H]')
table.column('Distance to the Sun', width=150, anchor='center')
table.column('[Fe/H]', width=50, anchor='center')    
table.column('E(B-V)', width=50, anchor='center')    

canvas.grid(column=0, row=5, columnspan=4)

def obtener_valor(event):
    global nombre
    global lienzo
    seleccion = table.focus()  # Obtener el item seleccionado
    nombre = table.item(seleccion)['text'] # Obtener el valor de la primera columna
    print("Valor seleccionado:", nombre)
    
    cat = 3 
    ext = 4 
    mul_ex = 2
    mul_ey = 2
    bn = 100  #bin del ajuste
    if 'lienzo' in globals():
        lienzo.get_tk_widget().pack_forget()
        lienzo.get_tk_widget().destroy()
    data_vasi = np.loadtxt(r'./Datos/Cúmulos_Vasiliev.csv', delimiter=';', dtype=np.dtype('U19'))
    Caracteristicas = np.loadtxt(directoriodata + r'/GCs(' + str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ').csv', delimiter=';', dtype=str)

    metalicidad = os.listdir(r'./Datos/Isocronas BASTI (elegidos)/')
    metalicidad_N = np.zeros(len(metalicidad), dtype=np.dtype('U25'))

    def gauss(X, x_0, y_0, sigma_x, sigma_y, a):
        x,y = X
        f = a*np.e**(-(((x-x_0)**2)/(2*sigma_x**2) + ((y-y_0)**2)/(2*sigma_y**2)))
        return f

    def elipse(x,y,a,b, h, k):
        return ((x-h)/a)**2 + ((y-k)/b)**2

    for i in range(len(metalicidad)):
          if 'm' in metalicidad[i]:
              metal_letras = list(metalicidad[i])
              metalicidad_N[i] = '-' +  metal_letras[4] + '.' + metal_letras[5] + metal_letras[6]
          elif 'p' in metalicidad[i]:
              metal_letras = list(metalicidad_N[i])
              metalicidad_N[i] = '+' + metal_letras[4] + '.' + metal_letras[5] + metal_letras[6]
          
          else:
              print('error')
              break
          
    repetidos = np.array([])
    for i in range(len(Caracteristicas[:,1])):
        if nombre in Caracteristicas[i,1]:
            indicek = i
            break

            
    print('----------------------------------------')
    

    print('Datos:')
    nombre = Caracteristicas[indicek,1]
    distancia = Caracteristicas[indicek,2]  #en kpc de Vasiliev
    extincion = Caracteristicas[indicek,ext]   
    metalicidad = list(Caracteristicas[indicek,cat])
    if len(metalicidad) == 4:
        metal = str(metalicidad[1]) + str(metalicidad[3]) + str(0) 
        metal_num = '-' + str(metalicidad[1]) + '.' + str(metalicidad[3]) + str(0) 
    else:
        metal = str(metalicidad[1]) + str(metalicidad[3]) + str(metalicidad[4]) 
        metal_num = '-' + str(metalicidad[1]) + '.' + str(metalicidad[3]) + str(metalicidad[4]) 
        
    directorio = r'Datos/Isocronas BASTI (elegidos)/' + 'FEHm' + metal + '/'
    archivos = os.listdir(directorio)
    for i in range(len(archivos)):
        os.rename(directorio + archivos[i], directorio + list(archivos[i])[0] + list(archivos[i])[1] +  list(archivos[i])[2] + list(archivos[i])[3]  +'0 Myr.txt')
    
    archivos = os.listdir(directorio)
    
    
    
    EBV = float(extincion)
    distance = float(distancia)*1000 #en pc
    memberprop_limit = 0.90

    print(nombre)
    print('[Fe/H] = ' + metal_num)
    print('Distancia al cúmulo: ' + distancia + ' [kpc]')
    print('Extinción NED: ' + str(EBV))
    
    directorios = os.listdir(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/')
    for i in range(len(directorios)):
        if nombre in directorios[i]:
            nombre = directorios[i].replace(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/', '')
            
    data = np.loadtxt(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/' + nombre)
    nombre = nombre.replace('.txt', '')
    
    memberprop = data[:,16]
    mp_index = np.zeros(len(memberprop), dtype=bool)
    for i in range(len(memberprop)):
        if memberprop[i]>memberprop_limit:
            mp_index[i] = True 
    #SE ELIMINAN LAS ESTRELLAS MÁS CERCANAS AL CENTRO DEL CÚMULO
    x = data[:,3]*60   #Posición centrado en el cúmulo(X) en arcmin
    y = data[:,4]*60   #Posición centrado en el cúmulo (Y) en arcmin


    ra = data[:,1]  #Ascensión recta
    dec = data[:,2] #Declinación
    
    
    el = memberprop>memberprop_limit
    xg = np.linspace(np.min(ra[el]), np.max(ra[el]), bn)
    yg = np.linspace(np.min(dec[el]), np.max(dec[el]), bn)
    xg, yg = np.meshgrid(xg, yg)
    H, xedges, yedges = np.histogram2d(ra[el], dec[el], bins=bn)
    p = [(np.max(ra[el])-np.min(ra[el]))/2 + np.min(ra[el]),(np.max(dec[el])-np.min(dec[el]))/2 + np.min(dec[el]),0.1,0.1,np.max(H)]
    popt, cov = curve_fit(gauss, (xg.flatten(), yg.flatten()), H.flatten(), p0=p)

     
    descartes_index = np.zeros(len(memberprop), dtype=bool)
    eleccion_index = np.zeros(len(memberprop), dtype=bool)
    
    a = mul_ex*popt[2]
    b = mul_ey*popt[3]
    for i in range(len(memberprop)):
        if elipse(ra[i], dec[i], a,b, popt[0], popt[1])<1:
            descartes_index[i] = True
        elif memberprop[i]>memberprop_limit:
            eleccion_index[i] = True
    
    
    gid = data[:,0] #GAIA ID
    
    ra = data[:,1]  #Ascensión recta
    dec = data[:,2] #Declinación
    
    x = data[:,3]   #Posición centrado en el cúmulo(X) en grados
    y = data[:,4]   #Posición centrado en el cúmulo (Y) en grados
    
    plx = data[:,5] #Paralaje corregido con zero-point offset usando Lindegren
    plxe = data[:,8]#Error en paralaje
    
    pmra = data[:,6]#Movimiento propio en la direción ra
    pmdec = data[:,7]#Movimiento propio en la dirección dec
    pmrae = data[:,9]#Error en mp en ra
    pmdece = data[:,10]#Error en mp en dec
    pmcorr = data[:,11]#Coeficiente de correlación entre los 2 errores de mp
    
    g_mag = data[:,12]#Magnitud en la banda G
    bp_rp = data[:,13]#BP-RP
    Sigma = data[:,14]#Densidad de fuentes (estrellas/arcmin^2)
    qflag = data[:,15]#0/1: 6/5 parametros; 0/2: estrellas que pasaron los filtros/estrellas que no
    memberprop = data[:,16]
    
    # Colour--Teff relation determined from Gaia DR2 data --> Computed using color_teff_GDR2.ipynb
        

    #Corrección de magnitudes
    ebv_array=np.ones(np.size(g_mag))*EBV
    distance_array = np.ones(np.size(g_mag))*distance

    poly = np.array([62.55257114, -975.10845442, 4502.86260828, -8808.88745592, 10494.72444183])
    Teff = np.poly1d(poly)(bp_rp) / 1e4

    # We enter in an iterative process of 10 steps to refine the AG and E_BP_RP computation. 'Cause we do not have intrinsic colour.
    for i in range(10):
        E_BP_RP = (-0.0698 + Teff*(3.837-2.530*Teff)) * EBV # A_BP - A_RP
        AG = (1.4013 + Teff*(3.1406-1.5626*Teff)) * EBV
        Teff = np.poly1d(poly)(bp_rp - E_BP_RP) / 1e4

    ABP = (1.7895 + Teff*(4.2355-2.7071*Teff)) * ebv_array
    ARP = (1.8593 + Teff*(0.3985-0.1771*Teff)) * ebv_array
    
    MG = g_mag - 5*(np.log10(np.ones(np.size(g_mag))*distance_array)) + 5 - AG
    COL_bp_rp = bp_rp - E_BP_RP
    
    
    stars = len(memberprop) #nº de estrellas
    difstars = stars-len(memberprop[mp_index]) #nº de estrellas eliminadas por la probabilidad
    startselec = len(memberprop[eleccion_index]) #nº de estrellas que cumplen los parametros
    #Plots
    #CMD E HISTOGRAMA
    fig = Figure(figsize=(12, 9))
    ax1 = fig.add_subplot(121)
    ax1.plot(COL_bp_rp[eleccion_index], MG[eleccion_index], '.', markersize=1.3, color='black', label='')
    fig.gca().add_patch(plt.Rectangle([0.5, 4.5], 0.5, -1.5, color='b', alpha=0.2))
    ax1.set_xlim(0, 2)
    if np.isnan(min(MG))==True:
        ax1.set_ylim(8,-3.5)
    else:
        ax1.set_ylim(8,min(MG)+2)
    ax2 = fig.add_subplot(122)
    ax2.plot(COL_bp_rp[eleccion_index], MG[eleccion_index], '.', markersize=1.3, color='black', label='')
    
    ax2.set_xlim(0.5, 1)
    ax2.set_ylim(4.5,3)
    
    for i in range(len(archivos)):
        data_iso = np.loadtxt(directorio + archivos[i])
        M_ini_iso = data_iso[:,0]
        M_fin_iso = data_iso[:,1]
        L_iso = data_iso[:,2]
        Te_iso = data_iso[:,3]
        MG_iso = data_iso[:,4]
        bp_iso = data_iso[:,5]
        rp_iso = data_iso[:,6]
        bp_rp_iso = bp_iso - rp_iso
        plt.subplot(121)
        ax1.plot(bp_rp_iso, MG_iso, label=archivos[i].replace('.txt',''))
        plt.subplot(122)
        ax2.plot(bp_rp_iso, MG_iso, label=archivos[i].replace('.txt',''))
    print(directorio + archivos[i])
    # plt.subplot(121)
    ax1.set_xlabel('G$_{BP}$-G$_{RP}$',fontsize=16)
    ax1.set_ylabel('M$_G$',fontsize=16)
    fig.suptitle(nombre + ', [Fe/H] = ' + metal_num + ', stars with mul_sigma <%.1f' %mul_ex,fontsize=16)
    ax1.legend(fontsize=14)
    
    
    ax2.set_xlabel('G$_{BP}$-G$_{RP}$',fontsize=16, color='b')
    ax2.set_ylabel('M$_G$',fontsize=16, color='b')
    ax2.legend(fontsize=14)
    
    for spine in ax2.spines.values():
        spine.set_edgecolor('b')
    ax2.xaxis.set_tick_params(color='blue', labelcolor='blue')
    ax2.yaxis.set_tick_params(color='blue', labelcolor='blue')
    plt.tight_layout()
    plt.close()
    
    
    
    
    lienzo = FigureCanvasTkAgg(fig, master=fr2)
    lienzo.draw()
    lienzo.get_tk_widget().pack()
table.bind("<<TreeviewSelect>>", obtener_valor)


#---------------------Button start manual fitting--------------------------------------------

def starter2():
    global nombre
    global lienzo
    global num_E
    global num_Fe
    global directorio_isocronas
    seleccion = table.focus()  # Obtener el item seleccionado
    nombre = table.item(seleccion)['text'] # Obtener el valor de la primera columna
    print("Valor seleccionado:", nombre)
    
    cat = 3 
    ext = 4 
    mul_ex = 2
    mul_ey = 2
    bn = 100  #bin del ajuste
    if 'lienzo' in globals():
        lienzo.get_tk_widget().pack_forget()
        lienzo.get_tk_widget().destroy()
    data_vasi = np.loadtxt(r'./Datos/Cúmulos_Vasiliev.csv', delimiter=';', dtype=np.dtype('U19'))
    Caracteristicas = np.loadtxt(directoriodata + r'/GCs(' + str(R_sunMAX) + 'kpc_' + 'E' + str(E_max) + '_FeMAX' + str(Fe_max) + '_FeMIN' + str(Fe_min) + ').csv', delimiter=';', dtype=str)
    
    metalicidad = os.listdir(r'./Datos/Isocronas BASTI (elegidos)/')
    metalicidad_N = np.zeros(len(metalicidad), dtype=np.dtype('U25'))
    
    def gauss(X, x_0, y_0, sigma_x, sigma_y, a):
        x,y = X
        f = a*np.e**(-(((x-x_0)**2)/(2*sigma_x**2) + ((y-y_0)**2)/(2*sigma_y**2)))
        return f
    
    def elipse(x,y,a,b, h, k):
        return ((x-h)/a)**2 + ((y-k)/b)**2
    
          
    repetidos = np.array([])
    for i in range(len(Caracteristicas[:,1])):
        if nombre in Caracteristicas[i,1]:
            indicek = i
            break
    
            
    print('----------------------------------------')
    
    
    print('Datos:')
    nombre = Caracteristicas[indicek,1]
    distancia = Caracteristicas[indicek,2]  #en kpc de Vasiliev
    extincion = num_E.get()
    metal_num = num_Fe.get()
    metalicidad_int = list(num_Fe.get())
    
    if len(metalicidad_int) == 4:
        metal = str(metalicidad_int[1]) + str(metalicidad_int[3]) + str(0) 
    elif len(metalicidad_int) == 2:
        metal = str(metalicidad_int[1]) + str(0) 
    else:
        metal = str(metalicidad_int[1]) + str(metalicidad_int[3]) + str(metalicidad_int[4]) 
        
    if not(os.path.exists('Datos/Isocronas BASTI (elegidos)/' + 'FEHm' + metal + '/')):
        for i in range(len(metalicidad)):
              if 'm' in metalicidad[i]:
                  metal_letras = list(metalicidad[i])
                  metalicidad_N[i] = '-' +  metal_letras[4] + '.' + metal_letras[5] + metal_letras[6]
              elif 'p' in metalicidad[i]:
                  metal_letras = list(metalicidad_N[i])
                  metalicidad_N[i] = '+' + metal_letras[4] + '.' + metal_letras[5] + metal_letras[6]
              
              else:
                  print('error')
                  break
        
        metal_num = metalicidad_N[np.where(np.min(np.abs(float(num_Fe.get())-metalicidad_N.astype(float)))==np.abs(float(num_Fe.get())-metalicidad_N.astype(float)))][0]
        metal = str(metal_num[1]) + str(metal_num[3]) + str(metal_num[4])
    directorio = r'Datos/Isocronas BASTI (elegidos)/' + 'FEHm' + metal + '/'
    archivos = os.listdir(directorio)
    for i in range(len(archivos)):
        os.rename(directorio + archivos[i], directorio + list(archivos[i])[0] + list(archivos[i])[1] +  list(archivos[i])[2] + list(archivos[i])[3]  +'0 Myr.txt')
    
    archivos = os.listdir(directorio)
    
    
    EBV = float(extincion)
    distance = float(distancia)*1000 #en pc
    memberprop_limit = 0.90
    print(nombre)
    print('[Fe/H] = ' + num_Fe.get())
    print('Distancia al cúmulo: ' + distancia + ' [kpc]')
    print('Extinción NED: ' + str(EBV))
    
    directorios = os.listdir(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/')
    for i in range(len(directorios)):
        if nombre in directorios[i]:
            nombre = directorios[i].replace(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/', '')
            
    data = np.loadtxt(r'Datos/Globular clusters from EDR3 (Eugene)/catalogues/' + nombre)
    nombre = nombre.replace('.txt', '')
    
    memberprop = data[:,16]
    mp_index = np.zeros(len(memberprop), dtype=bool)
    for i in range(len(memberprop)):
        if memberprop[i]>memberprop_limit:
            mp_index[i] = True 
    #SE ELIMINAN LAS ESTRELLAS MÁS CERCANAS AL CENTRO DEL CÚMULO
    x = data[:,3]*60   #Posición centrado en el cúmulo(X) en arcmin
    y = data[:,4]*60   #Posición centrado en el cúmulo (Y) en arcmin
    
    
    ra = data[:,1]  #Ascensión recta
    dec = data[:,2] #Declinación
    
    
    el = memberprop>memberprop_limit
    xg = np.linspace(np.min(ra[el]), np.max(ra[el]), bn)
    yg = np.linspace(np.min(dec[el]), np.max(dec[el]), bn)
    xg, yg = np.meshgrid(xg, yg)
    H, xedges, yedges = np.histogram2d(ra[el], dec[el], bins=bn)
    p = [(np.max(ra[el])-np.min(ra[el]))/2 + np.min(ra[el]),(np.max(dec[el])-np.min(dec[el]))/2 + np.min(dec[el]),0.1,0.1,np.max(H)]
    popt, cov = curve_fit(gauss, (xg.flatten(), yg.flatten()), H.flatten(), p0=p)
    
     
    descartes_index = np.zeros(len(memberprop), dtype=bool)
    eleccion_index = np.zeros(len(memberprop), dtype=bool)
    
    a = mul_ex*popt[2]
    b = mul_ey*popt[3]
    for i in range(len(memberprop)):
        if elipse(ra[i], dec[i], a,b, popt[0], popt[1])<1:
            descartes_index[i] = True
        elif memberprop[i]>memberprop_limit:
            eleccion_index[i] = True
    
    
    gid = data[:,0] #GAIA ID
    
    ra = data[:,1]  #Ascensión recta
    dec = data[:,2] #Declinación
    
    x = data[:,3]   #Posición centrado en el cúmulo(X) en grados
    y = data[:,4]   #Posición centrado en el cúmulo (Y) en grados
    
    plx = data[:,5] #Paralaje corregido con zero-point offset usando Lindegren
    plxe = data[:,8]#Error en paralaje
    
    pmra = data[:,6]#Movimiento propio en la direción ra
    pmdec = data[:,7]#Movimiento propio en la dirección dec
    pmrae = data[:,9]#Error en mp en ra
    pmdece = data[:,10]#Error en mp en dec
    pmcorr = data[:,11]#Coeficiente de correlación entre los 2 errores de mp
    
    g_mag = data[:,12]#Magnitud en la banda G
    bp_rp = data[:,13]#BP-RP
    Sigma = data[:,14]#Densidad de fuentes (estrellas/arcmin^2)
    qflag = data[:,15]#0/1: 6/5 parametros; 0/2: estrellas que pasaron los filtros/estrellas que no
    memberprop = data[:,16]
    
    # Colour--Teff relation determined from Gaia DR2 data --> Computed using color_teff_GDR2.ipynb
        
    
    #Corrección de magnitudes
    ebv_array=np.ones(np.size(g_mag))*EBV
    distance_array = np.ones(np.size(g_mag))*distance
    
    poly = np.array([62.55257114, -975.10845442, 4502.86260828, -8808.88745592, 10494.72444183])
    Teff = np.poly1d(poly)(bp_rp) / 1e4
    
    # We enter in an iterative process of 10 steps to refine the AG and E_BP_RP computation. 'Cause we do not have intrinsic colour.
    for i in range(10):
        E_BP_RP = (-0.0698 + Teff*(3.837-2.530*Teff)) * EBV # A_BP - A_RP
        AG = (1.4013 + Teff*(3.1406-1.5626*Teff)) * EBV
        Teff = np.poly1d(poly)(bp_rp - E_BP_RP) / 1e4
    
    ABP = (1.7895 + Teff*(4.2355-2.7071*Teff)) * ebv_array
    ARP = (1.8593 + Teff*(0.3985-0.1771*Teff)) * ebv_array
    
    MG = g_mag - 5*(np.log10(np.ones(np.size(g_mag))*distance_array)) + 5 - AG
    COL_bp_rp = bp_rp - E_BP_RP
    
    
    stars = len(memberprop) #nº de estrellas
    difstars = stars-len(memberprop[mp_index]) #nº de estrellas eliminadas por la probabilidad
    startselec = len(memberprop[eleccion_index]) #nº de estrellas que cumplen los parametros
    #Plots
    #CMD E HISTOGRAMA
    fig = Figure(figsize=(12, 9))
    ax1 = fig.add_subplot(121)
    ax1.plot(COL_bp_rp[eleccion_index], MG[eleccion_index], '.', markersize=1.3, color='black', label='')
    fig.gca().add_patch(plt.Rectangle([0.5, 4.5], 0.5, -1.5, color='b', alpha=0.2))
    ax1.set_xlim(0, 2)
    
    ax1.set_ylim(8,-3)
    
    ax2 = fig.add_subplot(122)
    ax2.plot(COL_bp_rp[eleccion_index], MG[eleccion_index], '.', markersize=1.3, color='black', label='')
    
    ax2.set_xlim(0.5, 1)
    ax2.set_ylim(4.5,3)
    
    for i in range(len(archivos)):
        data_iso = np.loadtxt(directorio + archivos[i])
        M_ini_iso = data_iso[:,0]
        M_fin_iso = data_iso[:,1]
        L_iso = data_iso[:,2]
        Te_iso = data_iso[:,3]
        MG_iso = data_iso[:,4]
        bp_iso = data_iso[:,5]
        rp_iso = data_iso[:,6]
        bp_rp_iso = bp_iso - rp_iso
        plt.subplot(121)
        ax1.plot(bp_rp_iso, MG_iso, label=archivos[i].replace('.txt',''))
        plt.subplot(122)
        ax2.plot(bp_rp_iso, MG_iso, label=archivos[i].replace('.txt',''))
    print(directorio + archivos[i])
    # plt.subplot(121)
    ax1.set_xlabel('G$_{BP}$-G$_{RP}$',fontsize=16)
    ax1.set_ylabel('M$_G$',fontsize=16)
    fig.suptitle(nombre + ', [Fe/H] = ' + str(metal_num) + ', stars with mul_sigma <%.1f' %mul_ex,fontsize=16)
    ax1.legend(fontsize=14)
    
    
    ax2.set_xlabel('G$_{BP}$-G$_{RP}$',fontsize=16, color='b')
    ax2.set_ylabel('M$_G$',fontsize=16, color='b')
    ax2.legend(fontsize=14)
    
    for spine in ax2.spines.values():
        spine.set_edgecolor('b')
    ax2.xaxis.set_tick_params(color='blue', labelcolor='blue')
    ax2.yaxis.set_tick_params(color='blue', labelcolor='blue')
    plt.tight_layout()
    fig.savefig(directorio_isocronas + '/' +  nombre  + '(MANUAL_CMD_ISO_FE_' + str(metal_num) +'_E_' + str(EBV) + ' ).png', dpi=300)
    plt.close()
    
    
    
    
    lienzo = FigureCanvasTkAgg(fig, master=fr2)
    lienzo.draw()
    lienzo.get_tk_widget().pack()


icon_search2 = tk.PhotoImage(file='icon_new.png').subsample(-20, 20)
start2 = tk.Button(canvas, text='CMD with new values', width=120, height=60, command=starter2, image=icon_search2, compound=tk.TOP)
start2.grid(row=2, column=2, rowspan=2, pady=4)

#--------------------------------New values--------------------------------------

label_title2 = tk.Label(canvas, text='Manual fitting', font=Font(weight="bold", size=14))
label_title2.grid(row=1, column=0, sticky='e', columnspan=2)   
E = tk.Label(canvas, text='E(B-V) =')
E.grid(row=2, column=0, sticky='e')

num_E = tk.StringVar()
E_entry = tk.Entry(canvas, textvariable=num_E)
E_entry.grid(row=2, column=1)


Fe = tk.Label(canvas, text='[Fe/H] =')
Fe.grid(row=3, column=0, sticky='e',pady=4)

num_Fe = tk.StringVar()
Fe_entry = tk.Entry(canvas, textvariable=num_Fe)
Fe_entry.grid(row=3, column=1,pady=4)

root.mainloop()



