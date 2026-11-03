:- encoding(utf8).
% knowledge_base/kb.pl: la knowledge base di SO-VVF-NEXT, in un solo file.
% Contiene la conoscenza di dominio (fatti), le regole e la dichiarazione dei fatti di scenario,
% che src/prolog_engine.py asserisce per ogni scenario (incidenti, mezzi, registro, stime).
% I fatti di dominio devono coincidere con le costanti di src/modello_generativo.py e src/domain.py:
% lo verifica tests/test_prolog.py.

% teoria/2 e assumibile/2 sono definiti in due blocchi: abduzione e integrita delle assegnazioni.
:- discontiguous teoria/2.
:- discontiguous assumibile/1.

% --- Conoscenza di dominio ---

% Oltre questo piano serve l autoscala per soccorrere le persone.
soglia_piani_autoscala(4).

% Regola di gravità: punteggio = persone + velocità + [sostanze] + [vulnerabilità alta] + [adiacenti >= 2].
punti_persone(confermate, 2).
punti_persone(incerte, 1).
punti_persone(assenti, 0).
punti_velocita(alta, 2).
punti_velocita(media, 1).
punti_velocita(bassa, 0).
soglia_critico(5).
soglia_alto(3).

% --- Fatti di scenario, asseriti da Python ---

:- dynamic incidente/6.        % incidente(Id, Tipo, Piano, Sostanze, Persone, Meteo)
:- dynamic registro/3.         % registro(Id, Vulnerabilita, NAdiacenti)
:- dynamic adiacenza/2.        % adiacenza(IdA, IdB), caricata nei due versi
:- dynamic mezzo/2.            % mezzo(Id, TipoMezzo)
:- dynamic velocita_stimata/2. % velocita_stimata(Id, Velocita)
:- dynamic gravita_stimata/2.  % gravita_stimata(Id, Gravita): stima esterna (ML, rete), se presente
:- dynamic assegnato/3.        % assegnato(IdMezzo, IdIncidente, Ruolo): proposta da verificare

% --- Campi non riconosciuti dal testo della chiamata ---
% Un campo che l'adattatore non riconosce arriva come l'atomo ignoto. Politica prudente: persone
% ignote valgono come confermate e sostanze ignote come presenti, perché sottostimare un incidente
% costa più che sovrastimarlo. Tipo e meteo ignoti non attivano regole specifiche.
persone_effettive(ignoto, confermate) :- !.
persone_effettive(P, P).

sostanze_presenti(si).
sostanze_presenti(ignoto).

% Il canadair non interviene dove ci sono (o potrebbero esserci) sostanze pericolose.
% copre/3 usa la negazione per fallimento: un mezzo copre un ruolo se non si riesce a dimostrare
% incompatibile/2. Il goal negato è sempre ground (M e I legati), quindi la NAF non si inceppa.
incompatibile(M, I) :-
    mezzo(M, canadair),
    incidente(I, _, _, S, _, _),
    sostanze_presenti(S).

% Gravita: stessi punteggi e soglie di modello_generativo.regola_gravita.
punteggio(Persone, Vel, Sost, Vuln, Adj, P) :-
    persone_effettive(Persone, Pe),
    punti_persone(Pe, Pp),
    punti_velocita(Vel, Pv),
    ( sostanze_presenti(Sost) -> Ps = 1 ; Ps = 0 ),
    ( Vuln == alta -> Pvu = 1 ; Pvu = 0 ),
    ( Adj >= 2 ->    Pa = 1 ; Pa = 0 ),
    P is Pp + Pv + Ps + Pvu + Pa.

gravita_da_punteggio(P, critico) :- soglia_critico(S), P >= S, !.
gravita_da_punteggio(P, alto)    :- soglia_alto(S),    P >= S, !.
gravita_da_punteggio(_, basso).

gravita_con(I, Vel, G) :-
    incidente(I, _, _, Sost, Pers, _),
    registro(I, Vuln, Adj),
    punteggio(Pers, Vel, Sost, Vuln, Adj, P),
    gravita_da_punteggio(P, G).

gravita(I, G) :-
    velocita_stimata(I, Vel),
    gravita_con(I, Vel, G).

% Propagazione del pericolo. La lista dei visitati impedisce i loop sui cicli (potatura dei cicli).
collegato(A, B) :- collegato(A, B, [A]).

collegato(A, B, _) :-
    adiacenza(A, B).
collegato(A, B, Visitati) :-
    adiacenza(A, C),
    \+ memberchk(C, Visitati),
    collegato(C, B, [C|Visitati]).

% Gravita di partenza: la stima esterna se c e (ML o rete bayesiana), altrimenti la regola
% applicata alla velocita stimata del fuoco. La scelta e fatta incidente per incidente: I e
% legato da incidente/6 prima del condizionale, quindi il risultato non dipende dall ordine dei
% fatti anche quando gravita_base/2 e chiamata con I libera (come in esposto/1).
gravita_base(I, G) :-
    incidente(I, _, _, _, _, _),
    (   gravita_stimata(I, G0)
    ->  G = G0
    ;   gravita(I, G)
    ).

esposto(I) :-
    gravita_base(J, critico),
    J \== I,
    collegato(I, J).

% Gravita operativa: un incidente alto esposto a uno critico e trattato come critico.
% Non sostituisce la gravita del dataset, che resta il target del ML.
% Forma deterministica: corretta anche quando G e gia legata (ad esempio G = alto).
gravita_operativa(I, G) :-
    gravita_base(I, G0),
    (   G0 == alto, esposto(I)
    ->  G = critico
    ;   G = G0
    ).

% Composizione della squadra. Ogni ruolo richiesto da un incidente va coperto da un mezzo distinto.
ruolo(I, estinzione) :-
    incidente(I, _, _, _, _, _).
ruolo(I, rinforzo) :-
    gravita_operativa(I, critico).
ruolo(I, soccorso_quota) :-
    incidente(I, Tipo, Piano, _, Persone, _),
    Tipo \== bosco,
    soglia_piani_autoscala(S),
    Piano >= S,
    persone_effettive(Persone, Pe),
    persone_da_soccorrere(I, Pe).
ruolo(I, decontaminazione) :-
    incidente(I, _, _, S, _, _),
    sostanze_presenti(S).

% Persone incerte in quota: si manda l autoscala solo se la gravita non e bassa.
persone_da_soccorrere(_, confermate).
persone_da_soccorrere(I, incerte) :-
    gravita_operativa(I, G),
    G \== basso.

squadra_richiesta(I, Ruoli) :-
    findall(R, ruolo(I, R), Ruoli).

% Quali tipi di mezzo possono coprire un ruolo.
adatto(autopompa, estinzione, _).
adatto(autopompa, rinforzo, _).
adatto(canadair, estinzione, I) :- incidente(I, bosco, _, _, _, _).
adatto(canadair, rinforzo, I) :- incidente(I, bosco, _, _, _, _).
adatto(autoscala, soccorso_quota, _).
adatto(nucleo_nbcr, decontaminazione, _).

copre(M, I, R) :-
    ruolo(I, R),
    mezzo(M, T),
    adatto(T, R, I),
    \+ incompatibile(M, I).

mezzi_adatti(I, R, Lista) :-
    findall(M, copre(M, I, R), Lista).

% Un mezzo e idoneo per un incidente se copre almeno uno dei suoi ruoli.
idoneo(M, I) :-
    mezzo(M, _),
    incidente(I, _, _, _, _, _),
    once(copre(M, I, _)).

mezzi_idonei(I, Lista) :-
    findall(M, idoneo(M, I), Lista).

% Spiegazioni: perche un ruolo e richiesto.
motivo(I, estinzione, incendio_segnalato) :-
    ruolo(I, estinzione).
motivo(I, rinforzo, gravita_critica) :-
    gravita_base(I, critico).
motivo(I, rinforzo, esposto_a(J)) :-
    gravita_base(I, alto),
    gravita_base(J, critico),
    J \== I,
    collegato(I, J).
motivo(I, soccorso_quota, persone_in_quota(Persone, Piano)) :-
    ruolo(I, soccorso_quota),
    incidente(I, _, Piano, _, Persone, _).
motivo(I, decontaminazione, sostanze_pericolose) :-
    ruolo(I, decontaminazione).

% --- Ragionamento con assunzioni (dispense, sezioni 4.6 e 4.8) ---
% Una procedura top-down raccoglie gli atomi assumibili usati in una dimostrazione. La stessa
% procedura serve per l abduzione (dimostrare un osservazione) e per i conflitti (dimostrare falso).
% Il corpo di una clausola della teoria e una lista di atomi; call(G) delega un goal alla KB.
% Prolog riserva false/0, quindi l atomo speciale della contraddizione si chiama falso.

dimostra([], H, H).
dimostra([A|R], H0, H) :-
    passo(A, H0, H1),
    dimostra(R, H1, H).

passo(call(G), H, H) :- !, call(G).
passo(A, H, H) :- member(A, H), assumibile(A).                % gia assunto in questa dimostrazione
passo(A, H, [A|H]) :- assumibile(A), \+ memberchk(A, H).     % nuova assunzione
passo(A, H0, H) :- teoria(A, Corpo), dimostra(Corpo, H0, H).

% Gli insiemi di assunzioni si confrontano in forma canonica (ordinati, senza ripetizioni).
minimi(Insiemi, Minimi) :-
    sort(Insiemi, Distinti),
    exclude(non_minimo(Distinti), Distinti, Minimi).

non_minimo(Insiemi, H) :-
    member(H2, Insiemi),
    H2 \== H,
    subset(H2, H).

% Abducibili: la velocita del fuoco (mai osservata a runtime) e, se la chiamata non lo dice, il
% valore di persone e sostanze. Una spiegazione assume esattamente cio che manca.
assumibile(vel(I, V)) :-
    incidente(I, _, _, _, _, _),
    member(V, [alta, media, bassa]).
assumibile(pers(I, P)) :-
    incidente(I, _, _, _, ignoto, _),
    member(P, [confermate, incerte, assenti]).
assumibile(sost(I, S)) :-
    incidente(I, _, _, ignoto, _, _),
    member(S, [si, no]).

% Teoria della gravita in forma di clausole: persone e sostanze note sono fatti della chiamata.
teoria(gravita_osservata(I, G),
       [vel(I, V), pers(I, P), sost(I, S),
        call(( registro(I, Vuln, Adj), punteggio(P, V, S, Vuln, Adj, Pt), gravita_da_punteggio(Pt, G) ))]).
teoria(pers(I, P), [call(( incidente(I, _, _, _, P, _), P \== ignoto ))]).
teoria(sost(I, S), [call(( incidente(I, _, _, S, _, _), S \== ignoto ))]).

% Vincoli di integrita della teoria abduttiva: una variabile ha un solo valore per incidente.
% Servono quando le osservazioni sono piu di una, per esempio due fonti che stimano la gravita.
teoria(falso, [vel(I, A), vel(I, B), call(A @< B)]).
teoria(falso, [pers(I, A), pers(I, B), call(A @< B)]).
teoria(falso, [sost(I, A), sost(I, B), call(A @< B)]).

incoerente(H) :-
    teoria(falso, Corpo),
    vincolo_violato(Corpo, H).

% Un vincolo e violato se il suo corpo si dimostra usando solo assunzioni gia presenti in H.
vincolo_violato([], _).
vincolo_violato([call(G)|R], H) :- !, call(G), vincolo_violato(R, H).
vincolo_violato([A|R], H) :- member(A, H), vincolo_violato(R, H).

% Spiegazioni di un insieme di osservazioni: insiemi H di assumibili tali che la KB con H dimostra
% tutte le osservazioni (KB + H |= Oss), H e coerente (KB + H non dimostra falso) ed e minimo.
spiegazioni(Osservazioni, Spiegazioni) :-
    findall(HS,
            ( dimostra(Osservazioni, [], H), msort(H, HS), \+ incoerente(HS) ),
            Candidate),
    minimi(Candidate, Spiegazioni).

% Velocita del fuoco compatibili con una gravita osservata: quelle che compaiono in almeno una
% spiegazione minima della gravita.
spiegazioni_velocita(I, G, Lista) :-
    spiegazioni([gravita_osservata(I, G)], Hs),
    findall(V, ( member(H, Hs), memberchk(vel(I, V), H) ), Vs),
    sort(Vs, Lista).

% --- Integrita delle assegnazioni come diagnosi basata sulla consistenza (sezione 4.6) ---
% Ogni assegnazione proposta e un assumibile ass/3. I vincoli di integrita derivano falso da
% combinazioni di assegnazioni incompatibili; un conflitto e un insieme minimo di assegnazioni
% che, insieme alla KB, implica falso. Una proposta valida non ha conflitti.
% Un ruolo lasciato scoperto non e una violazione ma una richiesta di rinforzi (ruolo_scoperto/2).
assumibile(ass(M, I, R)) :- assegnato(M, I, R).

vincolo_integrita(doppia_assegnazione(M),
                  [ass(M, I1, R1), ass(M, I2, R2), call((I1, R1) @< (I2, R2))]).
vincolo_integrita(non_adatto(M, I, R),
                  [ass(M, I, R), call(\+ copre(M, I, R))]).
vincolo_integrita(ruolo_coperto_due_volte(I, R),
                  [ass(M1, I, R), ass(M2, I, R), call(M1 @< M2)]).

teoria(falso, Corpo) :- vincolo_integrita(_, Corpo).

conflitto(Motivo, C) :-
    vincolo_integrita(Motivo, Corpo),
    dimostra(Corpo, [], H),
    msort(H, C).

conflitti_minimi(Conflitti) :-
    findall(C, conflitto(_, C), Cs),
    minimi(Cs, Conflitti).

violazione(Motivo) :-
    conflitto(Motivo, _).

ruolo_scoperto(I, R) :-
    ruolo(I, R),
    \+ assegnato(_, I, R).

% Conteggi delle clausole di questo file, per la relazione.
kb_statistiche(NPred, NRegole, NClausole) :-
    findall(N-R,
            ( predicate_property(user:H, file(F)), file_base_name(F, 'kb.pl'),
              predicate_property(user:H, number_of_clauses(N)),
              N > 0,
              aggregate_all(count, (clause(user:H, B), B \== true), R) ),
            Coppie),
    length(Coppie, NPred),
    findall(N, member(N-_, Coppie), Ns), sum_list(Ns, NClausole),
    findall(R, member(_-R, Coppie), Rs), sum_list(Rs, NRegole).
