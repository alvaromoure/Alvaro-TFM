clear all, close all
warning('off')

sDirOrigen='/media/julian/DataDisk2/Datasets/HM_Hospitales/24_04_2020/ANONYMIZED_NEW/';
%sDirOrigen='/media/julian/DataDisk2/Datasets/HM_Hospitales/20_07_2020/ANON_NEW/';

sDirDestino='/media/julian/DataDisk2/Datasets/HM_Hospitales/24_04_2020/ProcesadasJD/';
%sDirDestino='/media/julian/DataDisk2/Datasets/HM_Hospitales/20_07_2020/ProcesadasJD/';

sTabla='/media/julian/DataDisk2/Datasets/HM_Hospitales/24_04_2020/01.csv';
%sTabla='/media/julian/DataDisk2/Datasets/HM_Hospitales/20_07_2020/CDSL_01.csv'; 

% Construimos la lista de "COVID POSITVO" para filtrar 
tInfo=readtable( sTabla );

vIdx = strcmp( cellstr( tInfo.DIAGING_INPAT ), 'COVID19 - POSITIVO' );
%vIdx = strcmp( cellstr( tInfo.DIAGING_INPAT ), 'COVID CONFIRMADO' );

tInfo = tInfo( vIdx, : );
tIDs= tInfo.PATIENTID ;  
Sesiones = zeros(length(tIDs));
eDir=dir( fullfile(sDirOrigen,'**','*.DC3' ));
Edad = [];
Sexo = [];
for i=1:length(eDir)
    eAux= dir( strcat( eDir(i).folder, strcat(filesep(),'*.DC3')) );
    if length( eAux ) <= 6  % Si tiene m�s de tres cortes consideramos que ens un TAC
        sFicheroOrigen=strcat( eDir(i).folder, filesep(), eDir(i).name );
        info = dicominfo( sFicheroOrigen );
        %         if isfield( info, 'OverlayRows_0' ), rmfield( info, 'OverlayRows_0' ); end; 
        %         if isfield( info, 'OverlayColumns_0' ), rmfield( info, 'OverlayColumns_0' ); end; 
        %         if isfield( info, 'OverlayType_0' ), rmfield( info, 'OverlayType_0' ); end; 
        %         if isfield( info, 'OverlayBitAllocated_0' ), rmfield( info, 'OverlayBitAllocated_0' ); end; 
        %         if isfield( info, 'OverlayOrigin_0' ), rmfield( info, 'OverlayOrigin_0' ); end; 
        %         if isfield( info, 'OverlayBitPosition_0' ), rmfield( info, 'OverlayBitPosition_0' ); end; 
        
        % S�lo si es una CR. No cogemos CT. Y s�lo si es PA o AP
        if ( ( strcmp( info.Modality, 'CR' ) | strcmp( info.Modality, 'DX' ) ) & ...
                (strcmp( info.ViewPosition, 'PA' ) | strcmp( info.ViewPosition, 'AP' ) | ...
                strcmp( info.StudyDescription, 'RX TORAX AP' ) | strcmp( info.StudyDescription, 'RX Simple' ) ) )
            
            disp( strcat( 'Procesando: ', sFicheroOrigen, '. Con posicion: ', info.ViewPosition, '-' , info.StudyDescription ) );
           
         
            try
                % S�lo lo abrimos si est� en la lista de "COVID POSITVO"
                if any( tIDs, str2num(info.PatientID) )
                    [DICOM_image] = dicomread( info );

                    % Si est� definida, aplicamos la ventana
                    % Si est� definido el reescalado lo aplicamos

                    if isfield( info, 'RescaleSlope') & isfield( info, 'RescaleIntercept' )
                        DICOM_image = DICOM_image * info.RescaleSlope + info.RescaleIntercept;
                    end
                   
                    if isfield( info, 'WindowCenter') & isfield( info, 'WindowWidth' ),
                        DICOM_image( DICOM_image > ( info.WindowCenter(1) + info.WindowWidth(1) / 2 ) )= ( info.WindowCenter(1) + info.WindowWidth(1) / 2 );
                        DICOM_image( DICOM_image < ( info.WindowCenter(1) - info.WindowWidth(1) / 2 ) )= ( info.WindowCenter(1) - info.WindowWidth(1) / 2 );
                    end    

                    if strcmp( info.PhotometricInterpretation, 'MONOCHROME1' )
                        DICOM_image= imcomplement( DICOM_image );
                    end
                    
                    DICOM_image = uint16(65536 * mat2gray( DICOM_image )); % im=im2uint16( mat2gray( DICOM_image ) ); 
                    % imshow(DICOM_image,'DisplayRange', [] );
                    
                    indx = find(tIDs == str2num(info.PatientID));
                    Sesiones(indx) = Sesiones(indx) + 1;
                    
                    f = 0;
                    switch info.ViewPosition
                        case 'PA'
                            
                            ProyeccionRX = 'PA';
                            
                        case 'AP'
                            
                            ProyeccionRX = 'AP';
                            
                        otherwise
                    
                            f = 1;      
                            
                    end
                    
                    if f
                        switch info.StudyDescription
                            
                        case 'RX TORAX AP'
                            
                            ProyeccionRX = 'AP';
                        
                        case 'RX Simple'
                            
                            ProyeccionRX = 'PA';
                            
                        otherwise
                            ProyeccionRX = 'NI';
                        end
                    end
                    
                    
                    ind = find(tIDs == str2num(info.PatientID));
                    
                    Edad = [Edad,tInfo.EDAD_AGE(ind)];
                    temp = tInfo.SEXO_SEX(ind);
                    if strcmp(temp{1},'MALE')
                        Sexo = [Sexo,0];
                    else
                        Sexo = [Sexo,1];
                    end
                    
                    
                    %sName = strcat('HM__1__n__',ProyeccionRX,'__',info.Modality,'__',info.PatientID,'__',num2str(Sesiones(indx)),'__',eDir(i).name);
                    
                    %sFicheroDestino=strcat( sDirDestino, sName, '.png'  );
                    %imwrite( DICOM_image, sFicheroDestino, 'png' );  % png sin compresi�n, 16 bits
                    %% status=copyfile( sFicheroOrigen, sDirDestino ); 
                else
                    % disp( strcat( 'El paciente del fichero: ', sFicheroOrigen, ' no est� confirmado de COVID' ) );
                end;
            catch
                disp( strcat( 'Error procesando el fichero: ', sFicheroOrigen ) );
            end;
        elseif ~strcmp( info.Modality, 'CT' )
            disp( strcat( 'El fichero: -', sFicheroOrigen, '- corresponde a: -', info.Modality, '-', info.StudyDescription  ) );
        else
            % disp( strcat( 'El fichero: -', sFicheroOrigen, '- corresponde a: -', info.Modality, '-', info.StudyDescription  ) );
        end;
    end;
end;     
