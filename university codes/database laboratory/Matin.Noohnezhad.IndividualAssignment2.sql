--answer for the individual assignment 2 problem
use AdventureWorks
go

declare @year int
declare YearCursor cursor for 
select YEAR(RateChangeDate) [year]from HumanResources.EmployeePayHistory
group by YEAR(RateChangeDate)
order by YEAR(RateChangeDate)
open YearCursor
Fetch next from YearCursor into @year

while(@@FETCH_STATUS=0)
BEGIN
print 'Year: '+cast(@year as nvarchar)
print 'Row NO.	Employee Name	Employee ID	Pay rate'
print '=======	============= 	===========	=======	'

declare @rowNo int, @employeeName nvarchar(50),@employeeID int,@payRate money
declare SubCursor cursor for 
select top 10 ROW_NUMBER() OVER(ORDER BY pay.Rate desc) as rowNumber,personContact.FirstName+' '+ personContact.LastName,pay.EmployeeID,pay.Rate from HumanResources.EmployeePayHistory pay
join HumanResources.Employee emp on pay.EmployeeID = emp.EmployeeID
join Person.Contact personContact on personContact.ContactID = emp.ContactID
where YEAR(pay.RateChangeDate) = @year --order by pay.Rate desc

open SubCursor
Fetch next from SubCursor into @rowNo, @employeeName,@employeeID,@payRate

while(@@FETCH_STATUS=0)
BEGIN
print cast(@rowNo as nvarchar) +'    '+ cast(@employeeName  as nvarchar)+'    '+cast(@employeeID  as nvarchar)+'    '+cast(@payRate as nvarchar)
Fetch next from SubCursor into @rowNo, @employeeName,@employeeID,@payRate
END
Close SubCursor
Deallocate SubCursor

print '=========================================================================='
Fetch next from YearCursor into @year
END
Close YearCursor
Deallocate YearCursor
