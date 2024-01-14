%% Database Creation 
%Create JSON for requests
clinicalJSONfile="cases_request2.json";
%json è fondamentalmente un file di testo per cui può essere letto con le
%funzioni opportune di MATLAB
FID=fopen(clinicalJSONfile,"r");

text_param=fscanf(FID,"%s"); % s include tutti i caratteri + spazi bianchi 
%text_param contiente preciso preciso il contenuto del file json ed è più 
%comodo far così per non creare la stringa direttamente in Matlab ma
%importandola
fclose(FID);

%% CREATE REQUEST OBJECT
%Bisogna effettuare la domanda (query)per recuperare i dati associati al
%filtro impostato 
method=matlab.net.http.RequestMethod.POST;

contentTypeField=matlab.net.http.field.ContentTypeField('application/json');
%Nell'header andrà inserita l'info sul formato 
header=[contentTypeField]; %specifichiamo solo il campo sul formato ma ce ne sono diversi

body=matlab.net.http.MessageBody(jsondecode(text_param));
%la funzione jsondecode prende un testo formato json e restituisce un
%oggetto di tipo struct che contiene i campi definiti; da forma testuale a
%forma strutturata interpretabile da MATLAB.

% NB. method, header e body sono OGGETTI.
request_cases=matlab.net.http.RequestMessage(method,header,body);
destination="https://api.gdc.cancer.gov/cases/";
% OSS. Sono state scaricate SOLO le info relative ai "cases" e non ancora
% i file con cui creare la matrice features x values
uri=matlab.net.URI(destination);
[response_cases,completedrequest, history] = send(request_cases,uri);

%% ADD DATA TO DATABASE
% I data sono presenti nella response.
datasource='Connection_breast_cancer';
username='root'; password='maquandoarrivalamorte';

connection=database(datasource,username,password);
%ricorda di chiuderla dopo con close(connection)!!

if isopen(connection)==1
    fprintf("\nConnection has been successfully established")
else
    fprintf("\nError encountered while trying to establish a connection")
    fprintf("%s\n", connection.Message) 
    %stampa il messaggio per cui la connectione non è andata a buon fine
end

%% Interazione con la Base di Dati
%% Query utilizzando statements non-SELECT
% Use Database, insert into, alter, delete...

query = ['USE project_breast_cancer'];
try
    execute(connection,query);
    fprintf("\nQuery execution properly accomplished.")
catch e
    fprintf("\nQuery execution encountered an error.")
    fprintf("%s\n",e.Message)
end
%%
data=jsonencode(response_cases.Body.Data.data.hits);
data=jsondecode(strrep(data,'[]','NaN'));
%sostituiamo i dati mancanti/vuoti con NaN
%str_new=strrep(str,old,new)
%%

connection.AutoCommit="off";

% Add element to Subject Table
table_name1 = 'subjects';
table_name2 = 'demographic';
table_name3 = 'exposures';
table_name4 = 'diagnoses';
table_name5 = 'treatments';
% Utilizzo dei prepared statements
sqlQuery1 = ['INSERT INTO ' table_name1 ' VALUES(?, ?, ?, ?)'];
sqlQuery2 = ['INSERT INTO ' table_name2 ' VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'];
sqlQuery3 = ['INSERT INTO ' table_name3 ' VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)'];
sqlQuery4 = ['INSERT INTO ' table_name4 ' VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'];
sqlQuery5 = ['INSERT INTO ' table_name5 ' VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)'];

for i = 1 : size(data,1) %numero di righe di data

    try
      prepStat = databasePreparedStatement(connection, sqlQuery1);
      prepStat = bindParamValues(prepStat, [1:4], {string(data{i}.case_id), ...
          string(data{i}.submitter_id), string(data{i}.disease_type), string(data{i}.primary_site)});
      execute(connection, prepStat); 
      val1 = missing;
        val2 = missing;
        val3 = missing;
        val4 = missing;
        val5 = missing;
        val6 = missing;
        val7 = missing;
        val8 = missing;
        val9 = missing;
        val10 = missing;

        %isfield returns 1 if true vs 0 if false
        if (isfield(data{i}.demographic, "demographic_id"))
            val1 = string(data{i}.demographic.demographic_id);
        end
        if (isfield(data{i}.demographic, "race"))
            val2 = string(data{i}.demographic.race);
        end
        if (isfield(data{i}.demographic, "days_to_birth"))
            val3 = string(data{i}.demographic.days_to_birth);
        end
        if (isfield(data{i}.demographic, "ethnicity"))
            val4 = string(data{i}.demographic.ethnicity);
        end
        if (isfield(data{i}.demographic, "year_of_birth"))
            val5 = string(data{i}.demographic.year_of_birth);
        end
        if (isfield(data{i}.demographic, "vital_status"))
            val6 = string(data{i}.demographic.vital_status);
        end
        if (isfield(data{i}.demographic, "age_at_index"))
            val7 = string(data{i}.demographic.age_at_index);
        end
        if (isfield(data{i}.demographic, "year_of_death"))
            val8 = string(data{i}.demographic.year_of_death);
        end
        if (isfield(data{i}.demographic, "gender"))
            val9 = string(data{i}.demographic.gender);
        end
        if (isfield(data{i}, "case_id"))
            val10 = string(data{i}.case_id);
        end
        %questo procedimento va fatto per tutte le tabelle del database!!!
        prepStat = databasePreparedStatement(connection, sqlQuery2);
        prepStat = bindParamValues(prepStat, [1:10], {val1, val2, val3, ...
            val4, val5, val6, val7, val8, val9, val10});
        execute(connection, prepStat);
        
        for j = 1 : size(data{i}.exposures, 1)
            val1 = missing;
            val2 = missing;
            val3 = missing;
            val4 = missing;
            val5 = missing;
            val6 = missing;
            val7 = missing;
            val8 = missing;
            val9 = missing;
            
            if (isfield(data{i}.exposures(j), "exposure_id"))
                val1 = string(data{i}.exposures(j).exposure_id);
            end
            if (isfield(data{i}.exposures(j), "height"))
                val2 = string(data{i}.exposures(j).height);
            end
            if (isfield(data{i}.exposures(j), "weight"))
                val3 = string(data{i}.exposures(j).weight);
            end
            if (isfield(data{i}.exposures(j), "alcohol_history"))
                val4 = string(data{i}.exposures(j).alcohol_history);
            end
            if (isfield(data{i}.exposures(j), "cigarettes_per_day"))
                val5 = string(data{i}.exposures(j).cigarettes_per_day);
            end
            if (isfield(data{i}.exposures(j), "years_smoked"))
                val6 = string(data{i}.exposures(j).years_smoked);
            end
            if (isfield(data{i}.exposures(j), "bmi"))
                val7 =  string(data{i}.exposures(j).bmi);
            end
            if (isfield(data{i}.exposures(j), "alcohol_intensity"))
                val8 = string(data{i}.exposures(j).alcohol_intensity);
            end
            if (isfield(data{i}, "case_id"))
                val9 = string(data{i}.case_id);
            end
            prepStat = databasePreparedStatement(connection, sqlQuery3);
            prepStat = bindParamValues(prepStat, [1:9], {val1, val2, val3, ...
                val4, val5, val6, val7, val8, val9 });
            execute(connection, prepStat);
        end
        
        for j = 1 : size(data{i}.diagnoses, 1)
            
            val1 = missing;
            val2 = missing;
            val3 = missing;
            val4 = missing;
            val5 = missing;
            val6 = missing;
            val7 = missing;
            val8 = missing;
            val9 = missing;
            val0 = missing;
            val11 = missing;
            val12 = missing;
            val13 = missing;
            val14 = missing;
            val15 = missing;
            val16 = missing;
            val17 = missing;
            val18 = missing;
            
            if (isfield(data{i}.diagnoses(j), "diagnosis_id"))
                val1 = string(data{i}.diagnoses(j).diagnosis_id);
            end
            if (isfield(data{i}.diagnoses(j), "site_of_resection_or_biopsy"))
                val2 =  string(data{i}.diagnoses(j).site_of_resection_or_biopsy);
            end
            if (isfield(data{i}.diagnoses(j), "tissue_or_organ_of_origin"))
                val3 =  string(data{i}.diagnoses(j).tissue_or_organ_of_origin);
            end
            if (isfield(data{i}.diagnoses(j), "prior_treatment"))
                val4 = string(data{i}.diagnoses(j).prior_treatment);
            end
            if (isfield(data{i}.diagnoses(j), "age_at_diagnosis"))
                val5 = string(data{i}.diagnoses(j).age_at_diagnosis);
            end
            if (isfield(data{i}.diagnoses(j), "year_of_diagnosis"))
                val6 = string(data{i}.diagnoses(j).year_of_diagnosis);
            end
            if (isfield(data{i}.diagnoses(j), "ajcc_staging_system_edition"))
                val7 =  string(data{i}.diagnoses(j).ajcc_staging_system_edition);
            end
            if (isfield(data{i}.diagnoses(j), "ajcc_pathologic_t"))
                val8 = string(data{i}.diagnoses(j).ajcc_pathologic_t);
            end
            if (isfield(data{i}.diagnoses(j), "ajcc_pathologic_n"))
                val9 = string(data{i}.diagnoses(j).ajcc_pathologic_n);
            end
            if (isfield(data{i}.diagnoses(j), "ajcc_pathologic_m"))
                val10 =  string(data{i}.diagnoses(j).ajcc_pathologic_m);
            else
                nan='not reported';
                val10=string(nan);
            end
            if (isfield(data{i}.diagnoses(j), "tumor_stage"))
                val11 = string(data{i}.diagnoses(j).tumor_stage);
            end
            if (isfield(data{i}.diagnoses(j), "icd_10_code"))
                val12 = string(data{i}.diagnoses(j).icd_10_code);
            end
            if (isfield(data{i}.diagnoses(j), "morphology"))
                val13 = string(data{i}.diagnoses(j).morphology);
            end
            if (isfield(data{i}.diagnoses(j), "days_to_last_follow_up"))
                val14 = string(data{i}.diagnoses(j).days_to_last_follow_up);
            end
            if (isfield(data{i}.diagnoses(j), "synchronous_malignancy"))
                val15 = string(data{i}.diagnoses(j).synchronous_malignancy);
            end
            if (isfield(data{i}.diagnoses(j), "prior_malignancy"))
                val16 = string(data{i}.diagnoses(j).prior_malignancy);
            end
            if (isfield(data{i}.diagnoses(j), "primary_diagnosis"))
                val17 = string(data{i}.diagnoses(j).primary_diagnosis);
            end
            if (isfield(data{i}, "case_id"))
                val18 = string(data{i}.case_id);
            end

            prepStat = databasePreparedStatement(connection, sqlQuery4);
            prepStat = bindParamValues(prepStat, [1:18], {val1, val2, val3, ...
                val4, val5, val6, val7, val8, val9, val10, val11, val12, ...
                val13, val14, val15, val16, val17, val18});
            execute(connection, prepStat);
            
            for k = 1 : size(data{i}.diagnoses(j).treatments, 1)
                
                val1 = missing;
                val2 = missing;
                val3 = missing;
                val4 = missing;
                val5 = missing;
                val6 = missing;
                val7 = missing;
                val8 = missing;
                val9 = missing;
                if (isfield(data{i}.diagnoses(j).treatments{k}, "treatment_id"))
                    val1 = string(data{i}.diagnoses(j).treatments{k}.treatment_id);
                end
                if (isfield(data{i}.diagnoses(j).treatments{k}, "treatment_intent_type"))
                    val2 = string(data{i}.diagnoses(j).treatments{k}.treatment_intent_type);
                end
                if (isfield(data{i}.diagnoses(j).treatments{k}, "treatment_or_therapy"))
                    val3 = string(data{i}.diagnoses(j).treatments{k}.treatment_or_therapy);
                end
                if (isfield(data{i}.diagnoses(j).treatments{k}, "treatment_type"))
                    val4 = string(data{i}.diagnoses(j).treatments{k}.treatment_type);
                end
                if (isfield(data{i}.diagnoses(j).treatments{k}, "therapeutic_agents"))
                    val5 = string(data{i}.diagnoses(j).treatments{k}.therapeutic_agents);
                end
                if (isfield(data{i}.diagnoses(j).treatments{k}, "treatment_effect"))
                    val6 = string(data{i}.diagnoses(j).treatments{k}.treatment_effect);
                end
                if (isfield(data{i}.diagnoses(j).treatments{k}, "treatment_outcome"))
                    val7 =  string(data{i}.diagnoses(j).treatments{k}.treatment_outcome);
                end
                if (isfield(data{i}.diagnoses(j).treatments{k}, "initial_disease_status"))
                    val8 = string(data{i}.diagnoses(j).treatments{k}.initial_disease_status);
                end
                if (isfield(data{i}.diagnoses(j), "diagnosis_id"))
                    val9 = string(data{i}.diagnoses(j).diagnosis_id);
                end
                
                prepStat = databasePreparedStatement(connection, sqlQuery5);
                prepStat = bindParamValues(prepStat, [1:9], {val1, val2, ...
                val3, val4, val5, val6, val7, val8, val9 });
            execute(connection, prepStat);
            end
        end

        commit(connection);
        fprintf("Esecuzione della query numero %d riuscita\n", i);
        %qui finiscono gli statements del blocco try
    catch e
        rollback(connection);
        fprintf("\nEsecuzione della numero %d query non riuscita", i);
        fprintf("\n%s", e.message);
        if strcmp(e.identifier, "database:database:JDBCDriverError") ...
                && contains(e.message, "Duplicate entry")
            %tf = strcmp(s1,s2) compares s1 and s2 and returns 1 (true) if 
            %the two are identical and 0 (false) otherwise.
            %Text is considered identical if the size and content of each are the same. 
            %The return result tf is of data type logical.
            fprintf("Stai inserendo due volte la stessa chiave\n");
        else
            fprintf("Altro errore\n");
        end
    end
end

%% Scarichiamo i dati di rnaSEQ dei soggetti presenti nel DB
% Selezioniamo i case_id dei soggetti

table= 'demographic';
query_demographic = ['SELECT * FROM ', table];

%results = fetch(conn,sqlquery) - returns all rows of data after executing
%the SQL statement sqlquery for the connection object. fetch imports data in batches.

try
    table_demographic=fetch(connection,query_demographic);
    %questo fornisce tutti i dati della tabella demographic, inclusi i
    %case_id
    fprintf("\nEsecuzione della query riuscita\n");
catch e
    fprintf("\nEsecuzione della query non riuscita\n");
    fprintf("\n%s", e.message);
end

RNAseqJSONfile="RNAseq_request_breast.json";
FID=fopen(RNAseqJSONfile,"r");
text_RNAseq=fscanf(FID,"%s"); 
fclose(FID);
raw_text=jsondecode(text_RNAseq);

%inseriamo i case_id raccolti con la query nella sezione filters della
%richiesta da inoltrare ad http
raw_text=jsondecode(text_RNAseq);
raw_text.filters.content(1).content.value=string(table_demographic.case_id);



%% Recuperare dei files ad http

method=matlab.net.http.RequestMethod.POST;
contentTypeField=matlab.net.http.field.ContentTypeField('application/json');
header=[contentTypeField];
body=matlab.net.http.MessageBody(raw_text);

% NB. method, header e body sono OGGETTI.
request_RNAseq=matlab.net.http.RequestMessage(method,header,body);
destination="https://api.gdc.cancer.gov/files/";

uri=matlab.net.URI(destination);
[RNAseq_response,completedrequest, history] = send(request_RNAseq,uri);

% Prendiamo MANUALMENTE i files con workflow HTseq - counts
%%
RNAseq=RNAseq_response.Body.Data.data.hits;
new_hits={};
nh=1;
for i=1:size(RNAseq,1)
    if RNAseq{i}.file_name(38:42) == 'htseq'
        new_hits{nh,1}=RNAseq{i};
        nh=nh+1;
    end
end
RNAseq_response.Body.Data.data.hits=new_hits;

%% Scarichiamo e unzippiamo tutti i files in RNAseq_response sotto il nome filename
clearvars -except connection RNAseq_response
path = "C:\Users\Floriana\Documents\UNI 2017-in corso\Biennale\BIOINFORMATICA AVANZATA\07-01-21\progetto\progetto 2\rna_seq_files_breast\";
%%
for i = 1 : size(RNAseq_response.Body.Data.data.hits,1)
    file_id = RNAseq_response.Body.Data.data.hits{i}.file_id;
    filename = RNAseq_response.Body.Data.data.hits{i}.file_name;
    
    fullURL = ['https://api.gdc.cancer.gov/data/' file_id];
    websave(strcat(path, filename), fullURL);

    gunzip(strcat(path, filename))
    delete (strcat(path, filename))
    
end

%% Process FILES
i = 1;
file_id = RNAseq_response.Body.Data.data.hits{i}.file_id;
filename = RNAseq_response.Body.Data.data.hits{i}.file_name;
filename = filename(1:end-3);
case_id = RNAseq_response.Body.Data.data.hits{i}.cases.case_id;

A = readtable(strcat(path, filename), 'FileType', 'text');
%%
B = startsWith(A.Var1,"_");
%returns 1 (true) if str starts with the specified pattern, and returns 0 (false) otherwise.

A(B,:) = []; %restituisce solo i valori 1 in B quindi quelli da cancellare

% Remove version from gene id (if necessary)- PLOT TWIST lo è
v = cellfun(@(x) strsplit(x, "."), A.Var1, 'UniformOutput',false);
v = cellfun(@(x) x{1}, v, 'UniformOutput',false);
A.Properties.RowNames = v;%A.Var1;
A(:,1) = [];

C = cell(1, size(RNAseq_response.Body.Data.data.hits,1));

rna_seq_table = table('Size', [size(A,1), 0]);
%To create variables only, without any rows, specify 0 as the first element of sz.

rna_seq_table.Properties.RowNames = A.Properties.RowNames;
rna_seq_table.(file_id) = A{:,1};
C{i} = case_id;

for i = 2 : size(RNAseq_response.Body.Data.data.hits,1)
    fprintf("Processing file %d of %d\n", i, size(RNAseq_response.Body.Data.data.hits,1));
    file_id = RNAseq_response.Body.Data.data.hits{i}.file_id;
    filename = RNAseq_response.Body.Data.data.hits{i}.file_name;
    filename = filename(1:end-3);
    case_id = RNAseq_response.Body.Data.data.hits{i}.cases.case_id;
    
    A = readtable(strcat(path, filename), 'FileType', 'text');
    B = startsWith(A.Var1,"_");
    A(B,:) = [];

    A.Properties.RowNames = A.Var1;
    A(:,1) = [];
    rna_seq_table.(file_id) = A{:,1};
    C{i} = case_id;
end

%% SELEZIONIAMO SOLO I GENI SECONDO LA LORO CORRELAZIONE NEI CASI DI CANCRO
FID = fopen("gene_consensus_breast.json", "r");
object = jsondecode(fscanf(FID, "%s"));
fclose(FID);

geneList = {object(:).gene_id};
reducedTableGenes = rna_seq_table(geneList, :);

%% Create clinical table
clinical_data = table(); tumor_stage={}; 
low=['stage i','stage ia','stage ib','stage ii','stage iia','stage iib'];
high=['stage iii','stage iiia','stage iiib','stage iiic','stage iv'];

for i=1:size(C,2)
    try
        sqlquery=['SELECT tumor_stage FROM diagnoses WHERE (case_id= "', C{i},...
            '" AND NOT (tumor_stage="not reported") AND NOT (tumor_stage="stage x"))'];
        [results, metadata] = fetch(connection, sqlquery); 
        fprintf("Esecuzione della query riuscita\n");
        tumor_stage{i}=results;
        if contains(low,table2cell(results)) 
            clinical_data = [clinical_data; table("low", 'VariableNames',...
                {'Score'}, 'RowNames', {reducedTableGenes.Properties.VariableNames{i}})];
        elseif contains(high,table2cell(results))
            clinical_data = [clinical_data; table("high", 'VariableNames', ...
                {'Score'}, 'RowNames', {reducedTableGenes.Properties.VariableNames{i}})];
        end
    catch e
        fprintf("Esecuzione della query non riuscita\n");
        fprintf("%s\n", e.message);
    end
end

%% Process table in case lack some informations
genData = reducedTableGenes(:,clinical_data.Properties.RowNames);
clinicalData = clinical_data;
%save('genData_melanomas.mat','genData')














