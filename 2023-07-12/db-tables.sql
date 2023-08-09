-- 系统参数表
CREATE TABLE params(
   lineNo         CHAR(100) PRIMARY KEY     NOT NULL,
   workPos           CHAR(100)    NOT NULL,
   longStdDistance    REAL     NOT NULL,
   longStdError       REAL     NOT NULL,
   shortStdDistance   REAL     NOT NULL,
   shortStdError       REAL     NOT NULL,
   konNadIP           CHAR(100),
   konNadPort         CHAR(16),
   laserPort          CHAR(100),
   laserBaudrate      CHAR(100),
   password           CHAR(100)
);
insert into params values("T01", "Line06", "205","10", "180","5", "169.254.49.20", "502", "COM9", "38400", "Hisense2020")