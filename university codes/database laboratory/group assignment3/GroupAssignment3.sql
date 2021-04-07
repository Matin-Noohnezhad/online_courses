use [The Best University]
go

----------------------------------function for the average of specify student in a semesters--------------------------------
CREATE FUNCTION getStudentSemesterAverage
(
@studentCode NVARCHAR(50)
)
RETURNS TABLE
AS
RETURN(
select semester.SemesterName,AVG(sLog.Score) as average from Student student
JOIN StudentLog sLog on student.PersonID = sLog.StudentID
JOIN Presentation presentation on presentation.ID = sLog.PresentationID
JOIN Semester semester on semester.ID = presentation.SemesterID
WHERE student.StudentCode = @studentCode
group by semester.SemesterName
)
GO

select * from getStudentSemesterAverage(36502)

-------------------------------------------------------------------------------------------------------
----------------------------------procedure for get details of failed students------------------------------

create procedure getDetailsOfPersonWhoFailedMoreThanOnce
(
@FieldName NVARCHAR(50)
)
as
select b.StudentCode,b.FailedTerms,YEAR(semes.StartDate) EnterYear from 
(
select a.StudentCode,count(a.average) as FailedTerms from (
select student.StudentCode,semester.SemesterName,AVG(sLog.Score) as average from Student student
JOIN StudentLog sLog on student.PersonID = sLog.StudentID
JOIN Presentation presentation on presentation.ID = sLog.PresentationID
JOIN Semester semester on semester.ID = presentation.SemesterID
JOIN Field field on field.ID = presentation.FieldID
where field.Name= @FieldName
group by student.StudentCode,semester.SemesterName) a
where a.average<50 
group by a.StudentCode
having count(a.average)>1) b
JOIN Student stu on stu.StudentCode = b.StudentCode
JOIN Registration regis on regis.StudentID = stu.PersonID
JOIN [Status] stat on stat.ID = regis.StatusID
JOIN SupervisorLog supLog on supLog.ID = regis.SupervisorLogID
JOIN Semester semes on semes.ID = supLog.SemesterID
JOIN Field field on field.ID = supLog.FieldID
where stat.[Description] = N'ورود' and field.Name = @FieldName
go

exec getDetailsOfPersonWhoFailedMoreThanOnce @FieldName = 'Computer'
go

create index StuIndex1  on Student(PersonID)
create index StuIndex2 on Student(StudentCode)
create index StuLogIndex1  on StudentLog(ID)
create index StuLogIndex2  on StudentLog(StudentID)
create index StuLogIndex3  on StudentLog(PresentationID)
create index PresIndex1  on Presentation(ID)
create index PresIndex2  on Presentation(FieldID)
create index PresIndex3  on Presentation(SemesterID)
create index SemesIndex1  on Semester(SemesterName)
create index SemesIndex2  on Semester(ID)
create index SemesIndex3  on Semester(SemesterName)
create index FieldIndex1 on Field(ID)
create index FieldIndex2  on Field(Name)
create index RegisIndex1  on Registration(ID)
create index RegisIndex2  on Registration(StudentID)
create index RegisIndex3  on Registration(SupervisorLogID)
create index RegisIndex4  on Registration(SemesterID)
create index SupLogIndex1  on SupervisorLog(ID)
create index StatIndex1  on [Status](ID)



