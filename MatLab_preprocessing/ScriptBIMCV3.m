clear all, close all
warning('off')

% sDirOrigen='/home/julian/Documents/Datasets/BIMCV-COVID19/BIMCV-COVID19/'; 
% sDirDestino='/home/julian/Documents/Datasets/BIMCV-COVID19/PreprocesadoJI/'; 

sDirOrigen='H:\COVID19\BIMCV-COVID19\BIMCV-COVID19\'; 
sDirDestino='c:\temp\rx3\'; 

eDir=dir( strcat( sDirOrigen , '**', filesep, '*.png' ) );

for i=1:length(eDir)
        sFilePNGOrigen=strcat( eDir(i).folder, filesep, eDir(i).name );
                
        sFileJSONOrigen=strrep( sFilePNGOrigen,'.png','.json'); 

        disp( strcat( int2str( i ), '_Procesando el fichero: ', sFilePNGOrigen ) );

        image=imread( sFilePNGOrigen );
        info = imfinfo( sFilePNGOrigen );

        % Si no existe el json especï¿½fico para este archivo tomamos por defecto el que haya en el directorio 
        % Esto es importante para ver la vaiable MONOCHORME y otras variables
        if  ~isfile( sFileJSONOrigen  ),  
            eDirAux1=dir( strcat( eDir(i).folder , '/*.json' ) );
            sFileJSONOrigen=strcat( eDirAux1(1).folder, filesep, eDirAux1(1).name );
        end; 
        
        val = jsondecode(fileread( sFileJSONOrigen ));

        % Si estï¿½ definido, aplicamos el reescalado 
        if isfield( val, 'x00281052') & isfield( val, 'x00281053' ),
            RescaleSlope=val.x00281053.Value;
            RescaleIntercept=val.x00281052.Value;
            image = image * RescaleSlope + RescaleIntercept;
        end;
        
        % Si estï¿½ definida, aplicamos la ventana
        if isfield( val, 'x00281050') & isfield( val, 'x00281051' ),
            WindowCenter=val.x00281050.Value;
            WindowWidth=val.x00281051.Value;
            image( image > ( WindowCenter(1) + WindowWidth(1) / 2 ) )= ( WindowCenter(1) + WindowWidth(1) / 2 );
            image( image < ( WindowCenter(1) - WindowWidth(1) / 2 ) )= ( WindowCenter(1) - WindowWidth(1) / 2 );
        end;

        % si es imagen de gris inversa, invertimos de nuevo
        if string( val.x00280004.Value ) ~= 'MONOCHROME2'
            image = imcomplement( image );
        end;
        
        tipo='NI';   % Buscamos PA o AP, y quitar LATERAL
        
        if  strfind( eDir(i).name, 'ap_' ), tipo='AP';
        elseif strfind( eDir(i).name, 'pa_' ), tipo='PA';
        elseif strfind( eDir(i).name, 'lateral_' ), tipo='LATERAL';
        elseif strfind( eDir(i).name, 'll_' ), tipo='LATERAL';
        elseif isfield( val, 'x00185101' ) & isfield( val.x00185101, 'Value' )
            tipo=string( val.x00185101.Value );  % Orientaciï¿½n de la imagen
        end;
        
        if strcmp( tipo, 'LATERAL' ) | strcmp( tipo, 'LL' )
            if isfield( val, 'x00081032' ) & isfield( val.x00081032.Value, 'x00080104' ) & isfield( val.x00081032.Value.x00080104, 'Value' )
                aux=string( val.x00081032.Value.x00080104.Value );  % Orientaciï¿½n de la imagen
                if strfind( aux, 'PA O AP' ), tipo = 'NI';, % Definitiamente no sabemos la orientación 
                elseif strfind( aux, 'PA' ), tipo = 'PA';,
                elseif strfind( aux, 'AP' ), tipo = 'AP';,
                elseif strfind( aux, 'PORTATIL' ), tipo = 'PO';,% Las portátiles deberían ser AP. Ponemos PO y los miramos a mano
                end;
           end;
        end; 
        
        if strfind( eDir(i).name, '_cr' ), maquina='CR';
        elseif strfind( eDir(i).name, '_dx' ), maquina='DX';
        end;
        
        if info.BitDepth > 16
            disp( 'Fichero con bitdepth > 16 bits ' );
        elseif  info.BitDepth < 16
            disp( 'Fichero con bitdepth < 16 bits ' );
        end;
        image = uint16(65536 * mat2gray( image ));
        % imshow(image,'DisplayRange', [] );
        
        if strfind( eDir(i).name, 'S0' )
            indice=strfind( eDir(i).name, 'S0' ) + 2;
            paciente= eDir(i).name( indice:indice+3 );
        end;     
        
        if strfind( eDir(i).name, 'E0' )
            indice=strfind( eDir(i).name, 'E0' ) + 2;
            sesion= eDir(i).name( indice:indice+3 );
        end; 
        
        exploracion='1'; 
        if strfind( eDir(i).name, 'acq-' )
            indice=strfind( eDir(i).name, 'acq-' ) + 4;
            exploracion= eDir(i).name( indice );
        elseif strfind( eDir(i).name, 'run-' )
            indice=strfind( eDir(i).name, 'run-' ) + 4;
            exploracion= eDir(i).name( indice );        
        end; 

        % Parseamos el nombre del fichero para ajustarlo al formato que hemos definido
        sFilePNGDestino=strcat( sDirDestino, 'BIMCV__1__n__', tipo, '__', paciente, '__', sesion, '__', exploracion, '__', maquina, '.png' );
        while isfile( sFilePNGDestino ), 
            exploracion = int2str( str2num( exploracion ) + 1 ); 
            sFilePNGDestino=strcat( sDirDestino, 'BIMCV__1__n__', tipo, '__', paciente, '__', sesion, '__', exploracion, '__', maquina, '.png' );
        end; 

        % ESta chapuza es para solventar un bug de Matlab que aparece con el imwrite y cadenas de caracteres
        if isstring( sFilePNGDestino )  
            sFilePNGDestino = char( sFilePNGDestino );
        end;

        % if ~isfile( sFilePNG ) & ~strcmp( tipo, 'LATERAL' ), 
        if isfile( sFilePNGDestino ), disp( 'Fichero sobreescrito' ); end;     
        imwrite(image, sFilePNGDestino, 'png' ); 
        
end;
